"""M5 执行器：按计划调用检测器并回收结构化结果。

对应开发文档
    §9.4 动态决策流程（调用检测器并判断置信度是否足够）、§8.1 统一接口、
    §2.3 动态选择必须可验证（记录 execution_result / latency）、§17.4 策略层指标。

职责
    1. 按 Plan.execution_order 依次调用检测器；
    2. 记录每次调用的时延、状态与结果，形成可追溯的执行轨迹；
    3. 支持提前终止（stop_condition 对应的置信度足够时不再追加调用）。

不做（边界）
    - 不选择策略（由 policy/planner 负责）；
    - 不融合结果、不做冲突判定（属 M6，§12）；
    - 不修改检测器输出（§2.4）。

输入 / 输出
    输入：Plan、窗口数据、Context、每个检测器的配置
    输出：DetectionResult 列表（及执行轨迹）

关键约束
    - 单个检测器异常不得中断整体流程：捕获异常并生成 status=ERROR 的结果（§9.5 原则 D、§12.4）；
    - 未在注册表中的检测器不得静默跳过，需以 ERROR/SKIPPED 结果显式记录（§2.3 可验证）；
    - 推理/检测耗时必须实测记录，供 §17.4 策略层指标与 §19.1 时延评估使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.agent.planner import Plan
from src.agent.tool_registry import DetectorRegistry
from src.detectors.base import AttackType, DetectionResult, DetectorStatus


@dataclass(slots=True)
class ExecutionTrace:
    """单次检测器调用的执行轨迹（§2.3 要求的可验证记录）。"""

    detector_id: str
    latency_ms: float
    status: str
    attack_type: int | None
    confidence: float | None
    detail: Mapping[str, Any] = field(default_factory=dict)


class Executor:
    """计划执行器（§9.4）。"""

    def __init__(self, registry: DetectorRegistry) -> None:
        """初始化执行器。

        Args:
            registry: 检测器注册表（§8.4）。
        """
        self.registry = registry

    def execute(
        self,
        plan: Plan,
        data: Any = None,
        context: Any = None,
        configs: Mapping[str, Mapping[str, Any]] | None = None,
        stop_on_confidence: float | None = None,
        traces: list[ExecutionTrace] | None = None,
    ) -> list[DetectionResult]:
        """按计划依次执行检测器。

        Args:
            plan: 检测计划（§9.3）。
            data: 待检测窗口数据。
            context: GNSS Context。
            configs: 检测器 id → 该检测器的配置映射。
            stop_on_confidence: 达到该置信度即停止追加调用（§9.4 终止分支）；
                ``None`` 表示执行完计划中的全部检测器。
            traces: 可选的轨迹收集列表；传入时会就地追加记录。

        Returns:
            按执行顺序排列的 DetectionResult 列表。

        Note:
            检测器抛出的 ``NotImplementedError`` 等异常会被捕获并转为 status=ERROR
            的结果，保证编排链路不中断（§9.5 原则 D 的回退语义）。
        """
        configs = configs or {}
        results: list[DetectionResult] = []

        for detector_id in plan.execution_order:
            started = time.perf_counter()
            try:
                detector = self.registry.get(detector_id)
                result = detector.run(data=data, context=context, config=configs.get(detector_id, {}))
            except Exception as exc:  # noqa: BLE001 - 编排层需吞掉单点失败并显式记录
                result = DetectionResult(
                    detector_id=detector_id,
                    attack_type=AttackType.NORMAL,
                    confidence=0.0,
                    evidence={"error": type(exc).__name__, "message": str(exc)},
                    status=DetectorStatus.ERROR,
                )
            finally:
                latency_ms = (time.perf_counter() - started) * 1000.0

            if result.latency_ms is None:
                result.latency_ms = latency_ms
            results.append(result)

            if traces is not None:
                traces.append(
                    ExecutionTrace(
                        detector_id=detector_id,
                        latency_ms=latency_ms,
                        status=result.status.value,
                        attack_type=int(result.attack_type),
                        confidence=float(result.confidence),
                        detail=dict(result.evidence),
                    )
                )

            if (
                stop_on_confidence is not None
                and result.status is DetectorStatus.OK
                and result.confidence >= stop_on_confidence
            ):
                break  # §9.4：置信度足够，停止追加检测（§9.5 原则 A 低成本优先）

        return results

    @staticmethod
    def summarize_latency(results: Sequence[DetectionResult]) -> Mapping[str, float]:
        """汇总时延统计，供 §17.4 策略层指标使用。

        Args:
            results: 检测结果序列。

        Returns:
            含 ``total_ms``、``max_ms``、``count`` 的映射。
        """
        latencies = [float(r.latency_ms or 0.0) for r in results]
        return {
            "total_ms": sum(latencies),
            "max_ms": max(latencies) if latencies else 0.0,
            "count": float(len(latencies)),
        }
