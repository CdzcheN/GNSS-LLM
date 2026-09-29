"""测试 src/agent：注册表、规划器（§8.3 矩阵）、Level 0/1 策略与执行器。

对应开发文档
    §8.4 Detector Registry、§9.3 决策输出、§9.4 流程、§9.5 四原则、§9.6 实现层级。

覆盖要点
    - 注册表拒绝重复注册、可按 id 取用、可导出工具视图；
    - Plan 自洽性校验；
    - 规划器严格遵循 §8.3 矩阵，并在无可用策略时退化到兜底检测器（§9.5 原则 D）；
    - 执行器隔离单点故障、支持置信度提前终止（§9.4）。
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from src.agent.executor import Executor
from src.agent.planner import FALLBACK_DETECTOR, Plan, Planner
from src.agent.policy import ContextRulePolicy, FixedPolicy
from src.agent.tool_registry import DetectorRegistry
from src.detectors.base import AttackType, BaseDetector, DetectionResult
from src.detectors.threshold import ThresholdDetector


class _BoomDetector(BaseDetector):
    """用于验证“单点故障不中断编排”的故障检测器。"""

    detector_id = "boom"

    def run(self, data: Any = None, context: Any = None, config: Mapping[str, Any] | None = None) -> DetectionResult:
        raise RuntimeError("模拟检测器故障")


class _OkDetector(BaseDetector):
    """返回高置信度结果的假检测器。"""

    detector_id = "ok"

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def run(self, data: Any = None, context: Any = None, config: Mapping[str, Any] | None = None) -> DetectionResult:
        return DetectionResult(self.detector_id, AttackType.JAMMING, self.confidence)


class DetectorRegistryTest(unittest.TestCase):
    """§8.4 注册表行为。"""

    def test_duplicate_registration_rejected(self) -> None:
        registry = DetectorRegistry()
        registry.register(ThresholdDetector())
        with self.assertRaises(ValueError):
            registry.register(ThresholdDetector())

    def test_non_detector_rejected(self) -> None:
        with self.assertRaises(TypeError):
            DetectorRegistry().register(object())  # type: ignore[arg-type]

    def test_unknown_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            DetectorRegistry().get("nope")

    def test_as_tools_exposes_tool_view(self) -> None:
        registry = DetectorRegistry()
        registry.register(ThresholdDetector())
        tools = registry.as_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "threshold")

    def test_by_context(self) -> None:
        registry = DetectorRegistry()
        registry.register(ThresholdDetector())
        self.assertEqual(registry.by_context("Q"), ["threshold"])
        self.assertEqual(registry.by_context("H"), [])


class PlanTest(unittest.TestCase):
    """Plan 自洽性（§9.3）。"""

    def test_execution_order_defaults_to_selected(self) -> None:
        plan = Plan(selected_detectors=("a", "b"))
        self.assertEqual(plan.execution_order, ("a", "b"))

    def test_order_must_subset_of_selected(self) -> None:
        with self.assertRaises(ValueError):
            Plan(selected_detectors=("a",), execution_order=("b",))


class PlannerTest(unittest.TestCase):
    """规划器遵循 §8.3 策略适用性矩阵。"""

    def test_matrix_routing(self) -> None:
        plan = Planner().plan_from_flags(["cn0_decreasing"], available=["cno", "satellite"])
        self.assertEqual(plan.selected_detectors, ("cno", "satellite"))
        self.assertTrue(plan.verification_required)
        self.assertIn("cn0_decreasing", plan.selection_reason)

    def test_high_uncertainty_prefers_deep_model(self) -> None:
        plan = Planner().plan_from_flags(["high_uncertainty"], available=["deep_temporal", "threshold"])
        self.assertIn("deep_temporal", plan.selected_detectors)

    def test_unknown_flag_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Planner().plan_from_flags(["no_such_state"])

    def test_fallback_when_nothing_available(self) -> None:
        plan = Planner().plan_from_flags(["pvt_abnormal"], available=[])
        self.assertEqual(plan.selected_detectors, (FALLBACK_DETECTOR,))
        self.assertTrue(plan.fallback_used)

    def test_budget_truncates_selection(self) -> None:
        plan = Planner().plan_from_flags(["cn0_decreasing"], budget={"max_detectors": 1})
        self.assertEqual(len(plan.selected_detectors), 1)


class PolicyTest(unittest.TestCase):
    """Level 0 / Level 1 策略行为（§9.6、§18.2 对照）。"""

    def test_fixed_policy_is_context_free(self) -> None:
        plan = FixedPolicy().select()
        self.assertEqual(plan.selected_detectors, (FALLBACK_DETECTOR,))
        self.assertFalse(plan.verification_required)

    def test_rule_policy_routes_by_context_flags(self) -> None:
        context = type("Ctx", (), {"flags": ("cn0_decreasing",)})()
        plan = ContextRulePolicy().select(context=context, available=["cno", "satellite"])
        self.assertEqual(plan.selected_detectors, ("cno", "satellite"))

    def test_rule_policy_without_flags_falls_back(self) -> None:
        context = type("Ctx", (), {"flags": ()})()
        plan = ContextRulePolicy().select(context=context)
        self.assertTrue(plan.fallback_used)
        self.assertEqual(plan.selected_detectors, (FALLBACK_DETECTOR,))


class ExecutorTest(unittest.TestCase):
    """执行器：故障隔离与提前终止（§9.4、§9.5 原则 D）。"""

    def _registry(self) -> DetectorRegistry:
        registry = DetectorRegistry()
        registry.register(_BoomDetector())
        registry.register(_OkDetector())
        registry.register(ThresholdDetector())
        return registry

    def test_detector_failure_is_isolated(self) -> None:
        plan = Plan(selected_detectors=("boom", "ok"), execution_order=("boom", "ok"))
        results = Executor(self._registry()).execute(plan, {}, None, {})
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status.value, "error")
        self.assertEqual(results[1].attack_type, AttackType.JAMMING)

    def test_missing_detector_recorded_as_error(self) -> None:
        plan = Plan(selected_detectors=("ghost",), execution_order=("ghost",))
        results = Executor(self._registry()).execute(plan, {}, None, {})
        self.assertEqual(results[0].status.value, "error")

    def test_stop_on_confidence_halts_follow_up(self) -> None:
        plan = Plan(selected_detectors=("ok", "threshold"), execution_order=("ok", "threshold"))
        results = Executor(self._registry()).execute(plan, {}, None, {}, stop_on_confidence=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].detector_id, "ok")

    def test_latency_is_recorded(self) -> None:
        plan = Plan(selected_detectors=("ok",), execution_order=("ok",))
        results = Executor(self._registry()).execute(plan, {}, None, {})
        self.assertIsNotNone(results[0].latency_ms)
        summary = Executor.summarize_latency(results)
        self.assertEqual(summary["count"], 1.0)


if __name__ == "__main__":
    unittest.main()
