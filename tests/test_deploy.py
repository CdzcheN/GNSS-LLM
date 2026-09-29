"""测试 src/deploy：时延测量、参数量预算与 INT8 精度损失判定。

对应开发文档
    §19.1 模型部署目标（参数量 / 时延 / 内存）、§1.4 G8 参数量 < 10⁶、
    §19.2 量化流程与精度损失目标（< 1%，不达标须如实报告）。

覆盖要点
    - 时延测量包含预热、返回分位数，并对非法参数报错；
    - 参数量统计与预算判定（边界值 10⁶ 必须判为超预算）；
    - 量化流程步骤顺序与 §19.2 一致；
    - 相对精度损失与目标判定，未达标时报告须显式标注“如实报告”。
"""

from __future__ import annotations

import unittest

from src.deploy.profiling import (
    DEFAULT_WARMUP,
    PARAMETER_BUDGET,
    check_parameter_budget,
    count_parameters,
    measure_latency,
    profile,
)
from src.deploy.quantize import (
    QUANTIZATION_STEPS,
    build_report,
    check_accuracy_drop,
    relative_drop,
)


class _Parameter:
    """最小参数替身：仅提供 numel()。"""

    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _FakeModel:
    """最小模型替身：仅提供 parameters()。"""

    def __init__(self, *counts: int) -> None:
        self._parameters = [_Parameter(count) for count in counts]

    def parameters(self) -> list[_Parameter]:
        return self._parameters


class LatencyTest(unittest.TestCase):
    """§19.1 推理时延测量。"""

    def test_measure_latency_returns_stats(self) -> None:
        stats = measure_latency(lambda: None, repeats=20, warmup=2)
        self.assertEqual(stats.count, 20)
        self.assertLessEqual(stats.min_ms, stats.median_ms)
        self.assertLessEqual(stats.median_ms, stats.max_ms)
        self.assertGreaterEqual(stats.mean_ms, 0.0)

    def test_warmup_is_counted_separately(self) -> None:
        calls = {"n": 0}

        def forward() -> None:
            calls["n"] += 1

        measure_latency(forward, repeats=5, warmup=3)
        self.assertEqual(calls["n"], 8)

    def test_invalid_arguments_rejected(self) -> None:
        with self.assertRaises(ValueError):
            measure_latency(lambda: None, repeats=0)
        with self.assertRaises(ValueError):
            measure_latency(lambda: None, warmup=-1)

    def test_default_warmup_is_positive(self) -> None:
        self.assertGreater(DEFAULT_WARMUP, 0)


class ParameterBudgetTest(unittest.TestCase):
    """§1.4 G8 参数量目标 < 10⁶。"""

    def test_count_parameters_sums_modules(self) -> None:
        self.assertEqual(count_parameters(_FakeModel(10, 20, 30)), 60)

    def test_count_parameters_rejects_non_model(self) -> None:
        with self.assertRaises(TypeError):
            count_parameters(object())

    def test_budget_boundary(self) -> None:
        self.assertTrue(check_parameter_budget(PARAMETER_BUDGET - 1)["ok"])
        self.assertFalse(check_parameter_budget(PARAMETER_BUDGET)["ok"])

    def test_negative_parameters_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_parameter_budget(-1)

    def test_profile_summarizes(self) -> None:
        report = profile(forward=lambda: None, model=_FakeModel(100), repeats=5, warmup=1)
        self.assertIsNotNone(report.latency)
        self.assertEqual(report.parameters, 100)
        self.assertTrue(report.within_budget)
        self.assertIn("parameters_million", report.to_dict())


class QuantizationTest(unittest.TestCase):
    """§19.2 量化流程与精度损失。"""

    def test_steps_follow_doc_order(self) -> None:
        self.assertEqual(
            QUANTIZATION_STEPS,
            (
                "fp32_train_and_validate",
                "evaluate_on_validation",
                "quantize_int8",
                "reevaluate_on_test",
                "compare_accuracy_drop",
            ),
        )

    def test_relative_drop(self) -> None:
        self.assertAlmostEqual(relative_drop(1.0, 0.99), 0.01, places=6)
        self.assertEqual(relative_drop(0.0, 0.5), 0.0)

    def test_target_check(self) -> None:
        self.assertTrue(check_accuracy_drop(1.0, 0.995))   # 0.5% ≤ 1%
        self.assertFalse(check_accuracy_drop(1.0, 0.98))   # 2% > 1%

    def test_report_marks_missing_target(self) -> None:
        report = build_report("macro_f1", 0.90, 0.85)
        self.assertFalse(report.within_target)
        self.assertIn("如实报告", report.notes)

    def test_report_within_target(self) -> None:
        report = build_report("macro_f1", 0.90, 0.899, size_before_mb=20.0, size_after_mb=5.0)
        self.assertTrue(report.within_target)
        self.assertAlmostEqual(report.compression_ratio or 0.0, 4.0, places=4)

    def test_negative_target_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_accuracy_drop(1.0, 0.9, target=-0.1)


if __name__ == "__main__":
    unittest.main()
