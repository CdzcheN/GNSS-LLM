"""M5 策略选择策略：按 §9.6 实现层级提供 Level 0 / Level 1，并预留 Level 2 / Level 3。

对应开发文档
    §9 智能体动态策略选择（9.1 模块定位、9.2 决策输入、9.4 动态决策流程、9.5 四原则、
    9.6 实现层级）、§8.3 策略适用性矩阵、§18.2/§18.3 消融实验。

职责
    1. 定义策略选择接口 Policy；
    2. 提供 Level 0 固定规则策略（对照基线）；
    3. 提供 Level 1 上下文规则路由策略（基于 §8.3 矩阵）；
    4. 为 Level 2（轻量学习选择器）与 Level 3（Agent + Tool Calling）预留实现位。

不做（边界）
    - 不执行检测、不融合结果、不判定事件（分别属 executor、fusion、event）；
    - 不修改任何检测标签或置信度（§2.4）；
    - 不在策略层引入未来信息，选择只依据当前与历史窗口（§16.3 防泄漏）。

输入 / 输出
    输入：Context、数据质量、历史检测结果、可用检测器与预算（§9.2）
    输出：Plan（§9.3）

关键约束
    - 每个策略都必须可被 §18.2 / §18.3 消融实验独立替换（固定策略 vs 动态策略选择）；
    - 选择理由必须写入 Plan.selection_reason（§2.3 动态选择必须可验证）；
    - Level 0 策略必须与兜底路径等价可用（§9.5 原则 D）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from src.agent.planner import FALLBACK_DETECTOR, Plan, Planner

#: 实现层级（§9.6）。用于实验记录与消融对比。
POLICY_LEVELS: Mapping[int, str] = {
    0: "固定规则",
    1: "上下文规则路由",
    2: "轻量学习策略选择器",
    3: "Agent + Tool Calling",
}


class Policy(ABC):
    """策略选择接口（§9.1、§9.2、§9.3）。"""

    #: 该策略对应的实现层级（§9.6）。
    level: int = 0

    @abstractmethod
    def select(
        self,
        context: Any = None,
        data_quality: float | None = None,
        previous_results: Sequence[Any] | None = None,
        available: Sequence[str] | None = None,
        budget: Mapping[str, float] | None = None,
    ) -> Plan:
        """产出检测计划。

        Args:
            context: 当前 GNSS Context（§7.2）。
            data_quality: 数据质量评分（§7.4）。
            previous_results: 之前的检测结果（用于追加验证，§9.4）。
            available: 可用检测器 id。
            budget: 资源预算。

        Returns:
            Plan。
        """
        raise NotImplementedError

    def describe(self) -> Mapping[str, Any]:
        """返回策略元信息，供 §20.3 实验记录使用。"""
        return {"policy": type(self).__name__, "level": self.level, "level_name": POLICY_LEVELS.get(self.level, "未知")}


class FixedPolicy(Policy):
    """Level 0：固定规则策略（§9.6、§18.2 的“固定策略”对照）。"""

    level = 0

    def __init__(self, detectors: Sequence[str] = (FALLBACK_DETECTOR,)) -> None:
        """初始化固定策略。

        Args:
            detectors: 固定使用的检测器序列。
        """
        self.detectors = tuple(detectors)

    def select(self, context=None, data_quality=None, previous_results=None, available=None, budget=None) -> Plan:
        """始终返回同一组检测器（作为 §18.2 对照基线）。"""
        selected = tuple(name for name in self.detectors if available is None or name in available)
        if not selected:
            selected = (FALLBACK_DETECTOR,)
        return Plan(
            selected_detectors=selected,
            execution_order=selected,
            verification_required=False,
            stop_condition=None,
            resource_budget=dict(budget or {}),
            selection_reason="Level 0 固定策略：不依赖 Context（§9.6）",
            fallback_used=selected == (FALLBACK_DETECTOR,),
        )


class ContextRulePolicy(Policy):
    """Level 1：上下文规则路由策略（§8.3 矩阵、§9.4 流程）。"""

    level = 1

    def __init__(self, planner: Planner | None = None) -> None:
        """初始化。

        Args:
            planner: 规划器；``None`` 时使用默认 §8.3 路由表。
        """
        self.planner = planner or Planner()

    def select(
        self,
        context: Any = None,
        data_quality: float | None = None,
        previous_results: Sequence[Any] | None = None,
        available: Sequence[str] | None = None,
        budget: Mapping[str, float] | None = None,
    ) -> Plan:
        """依据 Context 判据走 §8.3 矩阵生成计划。

        Args:
            context: GNSS Context；需提供 ``flags``（环境状态判据）字段，
                例如 ``{"spectrum_abnormal": True}``。判据本身由 Context 层计算，
                以保证“检测策略选择依赖可验证的上下文”（§2.3）。
            data_quality: 数据质量评分；过低时应退化为兜底策略。
            previous_results: 历史结果，用于追加验证判断。
            available: 可用检测器。
            budget: 资源预算。

        Returns:
            Plan。

        Note:
            当 Context 未提供 flags（例如 §18.1 的“无 Context”对照组）时，
            退化为兜底检测器，保证系统仍可运行（§9.5 原则 D）。
        """
        flags = list(getattr(context, "flags", ()) or ())
        if not flags:
            return Plan(
                selected_detectors=(FALLBACK_DETECTOR,),
                execution_order=(FALLBACK_DETECTOR,),
                verification_required=False,
                resource_budget=dict(budget or {}),
                selection_reason="Level 1 规则路由：无上下文判据，退化到兜底检测器（§9.5 原则 D）",
                fallback_used=True,
            )
        return self.planner.plan_from_flags(
            flags=flags, available=available, budget=budget
        )


class LearningPolicy(Policy):
    """Level 2：轻量学习策略选择器（§9.6，待实现）。"""

    level = 2

    def select(self, context=None, data_quality=None, previous_results=None, available=None, budget=None) -> Plan:
        """从数据中学习策略选择（待实现）。

        Raises:
            NotImplementedError: 需要先固化策略选择器的训练标签与评估协议（§18.2）。
        """
        raise NotImplementedError("TODO(§9.6 Level 2): 轻量学习策略选择器尚未实现")


class AgentPolicy(Policy):
    """Level 3：Agent + Tool Calling（§9.6，待实现）。"""

    level = 3

    def select(self, context=None, data_quality=None, previous_results=None, available=None, budget=None) -> Plan:
        """由智能体编排工具调用（待实现）。

        Raises:
            NotImplementedError: 需先完成工具注册表与有限步数约束（§9.6、§19.3）。
        """
        raise NotImplementedError("TODO(§9.6 Level 3): Agent 工具调用策略尚未实现")
