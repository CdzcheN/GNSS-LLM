"""M7 异常事件管理器：把逐秒判定转换为事件级记录并驱动状态机。

对应开发文档
    §13.1 事件定义、§13.2 告警合并（默认 30 s）、§13.3 事件状态机、
    §15.1 在线流程（更新 Event State）、§17.3 事件级指标。

职责
    1. 接收逐秒（或逐窗口）融合结果，按 §13.2 的合并窗口聚合为事件；
    2. 驱动 §13.3 状态机：NORMAL → SUSPECTED → CONFIRMED → ONGOING → RECOVERED；
    3. 为 §17.3 事件级指标保留必要的统计（样本数、起止时间、置信度）。

不做（边界）
    - 不做检测与融合（属 M4/M6）；
    - 不生成自然语言描述（属 M8）；
    - 状态只能由结构化检测逻辑产生，不得由 LLM 生成（§13.3、§2.4）。

输入 / 输出
    输入：时间戳（数值秒）、攻击类型、置信度、所调用的检测策略、关键证据
    输出：EventRecord（进行中或已结束）

关键约束
    - 合并窗口默认 30 s（§13.2），可配置以便做敏感性实验；
    - 事件状态迁移必须落在 §13.3 允许的路径内，非法迁移直接报错；
    - 事件编号必须稳定可复现（不可依赖随机数，§20.4）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.detectors.base import AttackType
from src.event.event_record import EventRecord, EventState

#: 默认连续告警合并窗口（秒，§13.2）。
DEFAULT_ALERT_MERGE_WINDOW_S: float = 30.0

#: 默认确认所需持续时间（秒）。§13.3 只规定状态顺序而未给门限，
#: 因此默认 0 表示“首次检出即确认”，门限留待实验确定（不臆造数值）。
DEFAULT_CONFIRM_AFTER_S: float = 0.0

#: 事件编号前缀（与 §14.2 示例 ``20260927_001`` 风格一致的序号部分）。
EVENT_ID_PREFIX: str = "EV"


@dataclass(slots=True)
class EventUpdate:
    """一次 ``update()`` 的结果（§15.1 的“更新 Event State”分支）。

    Attributes:
        closed: 本次调用结束的事件（已置为 RECOVERED，需落日志与告警）；无则 ``None``。
        current: 本次调用后的活动事件；无则 ``None``。
        opened_new: 本次调用是否新开了事件。

    Note:
        单独返回 ``closed`` 是必要的：攻击类型切换或超出合并窗口时，一次调用会
        “关旧开新”，若只返回活动事件，被关闭的事件将无人落盘（§14.4 C4 要求保留原始日志）。
    """

    closed: EventRecord | None = None
    current: EventRecord | None = None
    opened_new: bool = False


class EventManager:
    """事件状态机与告警合并（§13.2、§13.3）。"""

    def __init__(
        self,
        merge_window_s: float = DEFAULT_ALERT_MERGE_WINDOW_S,
        confirm_after_s: float = DEFAULT_CONFIRM_AFTER_S,
        event_id_prefix: str = EVENT_ID_PREFIX,
    ) -> None:
        """初始化事件管理器。

        Args:
            merge_window_s: 连续告警合并窗口（§13.2 默认 30 s）。
            confirm_after_s: 从 SUSPECTED 晋升为 CONFIRMED 所需的持续时长（秒）。
            event_id_prefix: 事件编号前缀。
        """
        self.merge_window_s = float(merge_window_s)
        self.confirm_after_s = float(confirm_after_s)
        self.event_id_prefix = event_id_prefix

        self._current: EventRecord | None = None
        self._finished: list[EventRecord] = []
        self._counter = 0

    # ------------------------------------------------------------------ 查询

    @property
    def current(self) -> EventRecord | None:
        """当前进行中的事件（无则返回 None）。"""
        return self._current

    @property
    def finished(self) -> tuple[EventRecord, ...]:
        """已结束的事件序列（按结束顺序）。"""
        return tuple(self._finished)

    # ------------------------------------------------------------------ 更新

    def update(
        self,
        timestamp_s: float,
        attack_type: AttackType,
        confidence: float,
        strategies: Iterable[str] = (),
        key_evidence: Mapping[str, Any] | None = None,
        detection_delay_s: float | None = None,
    ) -> EventUpdate:
        """按 §15.1 的逐秒流程更新事件状态。

        Args:
            timestamp_s: 当前窗口时间（数值秒）。
            attack_type: 融合后的判定类别（§1.3 三态）。
            confidence: 融合置信度。
            strategies: 本次判定所调用的检测策略（§13.1 Selected Strategies）。
            key_evidence: 关键证据（§13.1 Key Evidence）。
            detection_delay_s: 检测时延（§13.1 Detection Delay）。

        Returns:
            EventUpdate：本次调用结束的事件、调用后的活动事件与新开事件标记。

        Raises:
            ValueError: 置信度越界，或事件状态迁移违反 §13.3。
        """
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence 必须在 [0, 1] 内：{confidence!r}")
        timestamp_s = float(timestamp_s)

        if self._current is None:
            if attack_type is AttackType.NORMAL:
                return EventUpdate()
            opened = self._open(
                timestamp_s, attack_type, confidence, strategies, key_evidence, detection_delay_s
            )
            return EventUpdate(current=opened, opened_new=True)

        gap = timestamp_s - self._current.end_time
        same_attack = attack_type is self._current.attack_type

        # 仍在合并窗口内且类别一致 → 继续累积（§13.2）
        if same_attack and gap <= self.merge_window_s:
            self._merge(timestamp_s, confidence, strategies, key_evidence)
            return EventUpdate(current=self._current)

        # 否则先结束旧事件（§13.3 → RECOVERED），再视情况开启新事件
        closed = self._finalize()
        if attack_type is AttackType.NORMAL:
            return EventUpdate(closed=closed)
        opened = self._open(
            timestamp_s, attack_type, confidence, strategies, key_evidence, detection_delay_s
        )
        return EventUpdate(closed=closed, current=opened, opened_new=True)

    def close(self, timestamp_s: float | None = None) -> EventRecord | None:
        """显式结束当前事件（数据流结束时调用）。

        Args:
            timestamp_s: 结束时间；``None`` 表示沿用当前事件的 end_time。

        Returns:
            已结束的事件；无进行中事件时返回 ``None``。
        """
        if self._current is None:
            return None
        if timestamp_s is not None:
            self._current.end_time = max(float(timestamp_s), self._current.start_time)
        return self._finalize()

    # ------------------------------------------------------------------ 内部

    def _next_id(self) -> str:
        """生成确定性事件编号（不依赖随机数，§20.4）。"""
        self._counter += 1
        return f"{self.event_id_prefix}{self._counter:03d}"

    def _open(
        self,
        timestamp_s: float,
        attack_type: AttackType,
        confidence: float,
        strategies: Iterable[str],
        key_evidence: Mapping[str, Any] | None,
        detection_delay_s: float | None,
    ) -> EventRecord:
        """开启新事件，并按 §13.3 决定初始状态。"""
        state = EventState.SUSPECTED if self.confirm_after_s > 0 else EventState.CONFIRMED
        event = EventRecord(
            event_id=self._next_id(),
            start_time=timestamp_s,
            end_time=timestamp_s,
            attack_type=attack_type,
            confidence=float(confidence),
            detection_delay_s=detection_delay_s,
            selected_strategies=tuple(dict.fromkeys(strategies)),
            key_evidence=dict(key_evidence or {}),
            state=state,
            samples=1,
        )
        self._current = event
        return event

    def _merge(
        self,
        timestamp_s: float,
        confidence: float,
        strategies: Iterable[str],
        key_evidence: Mapping[str, Any] | None,
    ) -> None:
        """把新窗口并入当前事件（§13.2）。"""
        event = self._current
        assert event is not None  # 由调用方保证
        event.end_time = timestamp_s
        event.confidence = max(event.confidence, float(confidence))
        event.samples += 1
        event.selected_strategies = tuple(dict.fromkeys((*event.selected_strategies, *map(str, strategies))))
        if key_evidence:
            merged = dict(event.key_evidence)
            merged.update(key_evidence)
            event.key_evidence = merged

        # §13.3：确认后再有持续告警即进入 ONGOING
        if event.state is EventState.CONFIRMED:
            self._transition(event, EventState.ONGOING)
        elif event.state is EventState.SUSPECTED and event.duration_s >= self.confirm_after_s:
            self._transition(event, EventState.CONFIRMED)

    def _finalize(self) -> EventRecord:
        """结束当前事件并归档（§13.3 → RECOVERED）。"""
        event = self._current
        assert event is not None  # 由调用方保证
        self._transition(event, EventState.RECOVERED)
        self._finished.append(event)
        self._current = None
        return event

    @staticmethod
    def _transition(event: EventRecord, state: EventState) -> None:
        """执行状态迁移并校验是否符合 §13.3。

        Raises:
            ValueError: 迁移不在允许路径内。
        """
        if event.state is state:
            return
        if not event.can_transition_to(state):
            raise ValueError(f"非法事件状态迁移：{event.state.value} → {state.value}（§13.3）")
        event.state = state
