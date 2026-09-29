"""M5 规划器：把选定的检测策略组装为可执行计划。

对应开发文档
    §9.3 决策输出（selected_detectors / execution_order / verification_required /
    stop_condition / resource_budget）、§9.4 动态决策流程、§9.5 策略选择四原则、
    §8.3 策略适用性矩阵、§17.4 策略层指标。

职责
    1. 定义计划数据结构 Plan（字段与 §9.3 一一对应）；
    2. 依据 §8.3 策略适用性矩阵，把“环境状态”映射为首选 + 辅助检测器组合；
    3. 给出执行顺序、是否需要追加验证、终止条件与资源预算。

不做（边界）
    - 不直接执行检测器（由 executor 负责）；
    - 不修改检测结果（§2.4）；
    - 不做深度模型的训练或推理细节（属 S7）。

输入 / 输出
    输入：Context 派生的环境判据（flags）、数据质量、历史结果、资源预算
    输出：Plan

关键约束
    - 必须提供回退路径：计划中若首选检测器不可用，应能退化为固定阈值检测器（§9.5 原则 D）；
    - 低成本优先：默认先执行低开销检测器，深度模型仅在需要时进入（§9.5 原则 A/C）；
    - 计划必须可留痕：§2.3 要求记录 selection_reason，故 Plan 携带理由字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: §8.3 策略适用性矩阵：环境状态 → (首选策略, 辅助策略, 目的)。
#: 表中取值与开发文档 §8.3 完全一致，作为默认路由表；可在配置中覆盖。
DEFAULT_APPLICABILITY: Mapping[str, tuple[str, str, str]] = {
    "spectrum_abnormal": ("spectrum", "cno", "快速发现射频异常"),
    "cn0_decreasing": ("cno", "satellite", "判断压制影响"),
    "satellite_abnormal": ("satellite", "pvt", "判断卫星层异常"),
    "observation_inconsistent": ("observation", "deep_temporal", "识别潜在欺骗"),
    "pvt_abnormal": ("pvt", "observation", "从导航解算层验证"),
    "metrics_conflict": ("multi", "deep_temporal", "交叉验证"),
    "high_uncertainty": ("deep_temporal", "multi", "深入分析"),
}

#: 兜底检测器（§9.5 原则 D：Agent 异常时固定规则检测器独立运行）。
FALLBACK_DETECTOR: str = "threshold"


@dataclass(slots=True)
class Plan:
    """检测计划的决策输出（字段与 §9.3 一致）。

    Attributes:
        selected_detectors: 本次选中的检测器集合。
        execution_order: 执行顺序（低成本优先，§9.5 原则 A）。
        verification_required: 是否需要追加验证（§9.4 分支）。
        stop_condition: 提前终止条件描述，例如“置信度 >= 0.9 即停止追加”。
        resource_budget: 资源预算（时延/调用次数上限），对应 §9.3 resource_budget。
        selection_reason: 选择理由，供 §2.3 留痕与实验分析。
        fallback_used: 是否退化到兜底检测器。
    """

    selected_detectors: tuple[str, ...] = ()
    execution_order: tuple[str, ...] = ()
    verification_required: bool = False
    stop_condition: str | None = None
    resource_budget: Mapping[str, float] = field(default_factory=dict)
    selection_reason: str = ""
    fallback_used: bool = False

    def __post_init__(self) -> None:
        """保证执行顺序与选中集合一致，避免计划自相矛盾。"""
        if self.selected_detectors and not self.execution_order:
            self.execution_order = tuple(self.selected_detectors)
        unknown = set(self.execution_order) - set(self.selected_detectors)
        if unknown:
            raise ValueError(f"execution_order 含未选中的检测器：{sorted(unknown)}")


class Planner:
    """规则式规划器（§9.4、§9.6 Level 1）。"""

    def __init__(
        self,
        applicability: Mapping[str, tuple[str, str, str]] = DEFAULT_APPLICABILITY,
        fallback: str = FALLBACK_DETECTOR,
    ) -> None:
        """初始化规划器。

        Args:
            applicability: 策略适用性路由表（默认取 §8.3）。
            fallback: 兜底检测器 id（§9.5 原则 D）。
        """
        self.applicability = dict(applicability)
        self.fallback = fallback

    def plan_from_flags(
        self,
        flags: Sequence[str],
        available: Sequence[str] | None = None,
        budget: Mapping[str, float] | None = None,
        verification_margin: float = 0.2,
    ) -> Plan:
        """依据环境判据生成计划（§9.4 的确定性部分）。

        Args:
            flags: 命中的环境状态键（取自 ``DEFAULT_APPLICABILITY``）。
            available: 当前可用的检测器 id；``None`` 表示不做可用性过滤。
            budget: 资源预算（如 ``{"max_detectors": 2, "max_latency_ms": 50}``）。
            verification_margin: 触发追加验证的置信度余量（§12.4 冲突判据）。

        Returns:
            Plan。

        Raises:
            ValueError: flags 中含未定义的环境状态键。

        Note:
            置信度驱动的分支（§9.4 “结果置信度是否足够？”）在 executor 执行后由
            policy/fusion 判定并再次调用本方法生成追加计划，保持规划器无状态。
        """
        unknown = [flag for flag in flags if flag not in self.applicability]
        if unknown:
            raise ValueError(f"未定义的环境状态：{unknown}；可用 {sorted(self.applicability)}")

        selected: list[str] = []
        reasons: list[str] = []
        for flag in flags:
            primary, auxiliary, purpose = self.applicability[flag]
            for name in (primary, auxiliary):
                if name in ("multi",) or name in selected:
                    continue  # "multi" 表示多策略联合，由 policy 层展开（§8.3）
                if available is not None and name not in available:
                    continue
                selected.append(name)
                reasons.append(f"{flag}→{name}({purpose})")

        if not selected:
            selected = [self.fallback]
            reasons = [f"无可用策略，退化到兜底检测器 {self.fallback}（§9.5 原则 D）"]

        budget = dict(budget or {})
        max_detectors = int(budget.get("max_detectors", len(selected) or 1))
        truncated = selected[:max_detectors]

        return Plan(
            selected_detectors=tuple(truncated),
            execution_order=tuple(truncated),
            verification_required=len(truncated) > 1,
            stop_condition=f"首要检测器置信度 >= {1.0 - verification_margin:.1f} 时停止追加",
            resource_budget=budget,
            selection_reason="; ".join(reasons),
            fallback_used=truncated == [self.fallback],
        )
