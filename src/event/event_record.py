"""M7 事件数据结构与状态定义：把逐秒检测结果固化为事件级记录。

对应开发文档
    §13.1 事件定义（Event ID / Start Time / End Time / Duration / Attack Type /
    Confidence / Detection Delay / Selected Strategies / Key Evidence）、
    §13.2 告警合并、§13.3 事件状态机、§14.2 LLM 输入所需的事件字段。

职责
    1. 定义事件生命周期状态 EventState（§13.3 五态）；
    2. 定义事件记录 EventRecord（字段与 §13.1 一致）；
    3. 提供时间格式化，便于输出到 §14.2 的 LLM 输入与人工报告。

不做（边界）
    - 不做事件合并与状态迁移（属 event_manager）；
    - 不做持久化（属 event_logger）；
    - 不产生“未由结构化检测逻辑确认”的状态（§13.3 明确规定状态不得由 LLM 生成）。

输入 / 输出
    输入：由事件管理器填充
    输出：可 JSON 序列化的事件字典（to_dict）

时间表示
    为提高合并计算的可测性，事件时间以数值秒存储；``time_range_text()`` 提供
    ``HH:MM:SS`` 形式的展示文本（供 §14.2 LLM 输入与报告使用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping

from src.detectors.base import AttackType

#: 秒 → ``HH:MM:SS`` 的最大进制（一天 86400 秒）。
_SECONDS_PER_DAY: int = 86_400


class EventState(str, Enum):
    """事件状态机取值（§13.3）。

    状态只能由结构化检测逻辑产生，LLM 无权改写（§13.3、§2.4）。
    """

    NORMAL = "normal"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    ONGOING = "ongoing"
    RECOVERED = "recovered"


#: 状态机的允许迁移（§13.3：NORMAL → SUSPECTED → CONFIRMED → ONGOING → RECOVERED）。
ALLOWED_TRANSITIONS: Mapping[EventState, tuple[EventState, ...]] = {
    EventState.NORMAL: (EventState.SUSPECTED,),
    EventState.SUSPECTED: (EventState.CONFIRMED, EventState.RECOVERED),
    EventState.CONFIRMED: (EventState.ONGOING, EventState.RECOVERED),
    EventState.ONGOING: (EventState.RECOVERED,),
    EventState.RECOVERED: (EventState.SUSPECTED,),
}


def format_seconds(seconds: float) -> str:
    """把数值秒格式化为 ``HH:MM:SS``（供展示与 §14.2 输入使用）。

    Args:
        seconds: 数值秒（超过一天按一天取模）。

    Returns:
        ``HH:MM:SS`` 字符串。
    """
    total = int(seconds) % _SECONDS_PER_DAY
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


@dataclass(slots=True)
class EventRecord:
    """异常事件记录（字段与 §13.1 一致）。

    Attributes:
        event_id: 事件编号（§13.1 Event ID）。
        start_time: 起始时间（数值秒）。
        end_time: 结束时间（数值秒，进行中的事件随窗口推进）。
        attack_type: 事件对应的攻击类型（§1.3 三态）。
        confidence: 事件置信度，取成员窗口的融合置信度。
        detection_delay_s: 检测时延（§13.1 Detection Delay）。
        selected_strategies: 触发本事件的检测策略（§13.1 Selected Strategies）。
        key_evidence: 关键证据（§13.1 Key Evidence，取自各检测器 evidence）。
        state: 事件状态（§13.3）。
        samples: 合并到本事件的逐秒结果数量（用于事件级指标，§17.3）。
    """

    event_id: str
    start_time: float
    end_time: float
    attack_type: AttackType = AttackType.NORMAL
    confidence: float = 0.0
    detection_delay_s: float | None = None
    selected_strategies: tuple[str, ...] = ()
    key_evidence: Mapping[str, Any] = field(default_factory=dict)
    state: EventState = EventState.SUSPECTED
    samples: int = 1

    @property
    def duration_s(self) -> float:
        """事件持续时长（§13.1 Duration）。"""
        return max(0.0, float(self.end_time) - float(self.start_time))

    def can_transition_to(self, state: EventState) -> bool:
        """判断是否允许迁移到给定状态（§13.3）。

        Args:
            state: 目标状态。

        Returns:
            允许返回 True。
        """
        return state in ALLOWED_TRANSITIONS.get(self.state, ())

    def time_range_text(self) -> str:
        """返回 ``起始 ~ 结束`` 的展示文本（供 §14.2 LLM 输入与报告使用）。"""
        return f"{format_seconds(self.start_time)} ~ {format_seconds(self.end_time)}"

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化字典（字段含 §13.1 全部 9 项 + state + samples）。"""
        payload: dict[str, Any] = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["attack_type"] = int(self.attack_type)
        payload["attack_name"] = self.attack_type.name
        payload["state"] = self.state.value
        payload["duration_s"] = self.duration_s
        payload["start_time_text"] = format_seconds(self.start_time)
        payload["end_time_text"] = format_seconds(self.end_time)
        payload["selected_strategies"] = list(self.selected_strategies)
        payload["key_evidence"] = dict(self.key_evidence)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return (
            f"<EventRecord {self.event_id} {self.state.value} {self.attack_type.name} "
            f"{self.duration_s:.1f}s conf={self.confidence:.2f}>"
        )


def as_llm_input(event: EventRecord, data_quality: float | None = None) -> dict[str, Any]:
    """把事件转换为 §14.2 允许的 LLM 输入结构。

    Args:
        event: 事件记录。
        data_quality: 数据质量评分（§14.2 的 data_quality 字段）。

    Returns:
        字段严格落在 §14.2 白名单内的事件字典。

    Note:
        该函数是“结构化结果 → LLM 输入”的唯一出口，确保不夹带原始数据（§14.4 C1）；
        关键证据只保留 §14.2 白名单内的子字段，其余证据仍完整保存在原始日志中（§14.4 C4）。
    """
    # 延迟导入：对 §14.2 契约的依赖只出现在这个桥接函数内，避免 M7 → M8 的模块级耦合
    from src.llm.schema import ALLOWED_KEY_FEATURES

    key_features = {
        name: value
        for name, value in (event.key_evidence or {}).items()
        if name in ALLOWED_KEY_FEATURES
    }

    payload = {
        "event_id": event.event_id,
        "start_time": format_seconds(event.start_time),
        "end_time": format_seconds(event.end_time),
        "duration_s": event.duration_s,
        "state": event.attack_type.name.lower(),
        "confidence": float(event.confidence),
        "selected_strategies": list(event.selected_strategies),
        "key_features": key_features,
    }
    if event.detection_delay_s is not None:
        payload["detection_delay_s"] = float(event.detection_delay_s)
    if data_quality is not None:
        payload["data_quality"] = float(data_quality)
    return payload


#: 供调用方直接使用的字段名清单（§13.1）。
EVENT_FIELDS: tuple[str, ...] = (
    "event_id",
    "start_time",
    "end_time",
    "duration_s",
    "attack_type",
    "confidence",
    "detection_delay_s",
    "selected_strategies",
    "key_evidence",
)
