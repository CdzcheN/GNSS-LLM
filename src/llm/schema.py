"""M8 输入 / 输出 schema：把 §14.2 的结构化字段固化为可校验契约。

对应开发文档
    §14.2 LLM 输入、§14.3 LLM 输出、§14.4 约束 C2（事实优先）/C3（未知显式说明）、
    §18.8 LLM 模块验证（字段完整性、结构化字段与文本是否一致）。

职责
    1. 定义 LLM 允许接收的事件字段白名单与必需字段（§14.2）；
    2. 提供 ``validate_event`` 校验函数，拒绝越界输入（§14.4 C1 只读结构化结果）；
    3. 定义摘要输出记录结构（§14.4 C5）供归档。

不做（边界）
    - 不做字段的类型推断或自动补全（缺失必须显式暴露，§14.4 C3）；
    - 不允许通过额外字段把原始 GNSS 数据透传给模型（§1.5、§14.4 C1）；
    - 不判定检测标签的正确性（属检测与融合层）。

输入 / 输出
    输入：事件字典
    输出：校验结果（缺失字段列表 / 未识别字段列表）

关键约束
    - 白名单必须是固定集合：新增字段需同步更新本文档与提示词版本（§28）；
    - 校验只报告问题，不修改输入字典（保持原始日志，§14.4 C4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: 必需字段（§14.2 JSON 示例中承接核心事实的字段）。
REQUIRED_FIELDS: tuple[str, ...] = (
    "event_id",
    "start_time",
    "end_time",
    "state",
    "confidence",
)

#: 允许出现的字段白名单（§14.2）。未在白名单中的键会被报告为越界。
ALLOWED_FIELDS: tuple[str, ...] = (
    "event_id",
    "start_time",
    "end_time",
    "duration_s",
    "state",
    "confidence",
    "selected_strategies",
    "detection_delay_s",
    "key_features",
    "data_quality",
)

#: ``key_features`` 允许的子字段（§14.2 示例）。
ALLOWED_KEY_FEATURES: tuple[str, ...] = (
    "cn0_change",
    "satellite_change",
    "pdop_change",
    "position_error",
)

#: 事件生命周期状态（§13.3 事件状态机）。
EVENT_LIFECYCLE_STATES: tuple[str, ...] = (
    "normal",
    "suspected",
    "confirmed",
    "ongoing",
    "recovered",
)

#: 攻击类型取值（§1.3 三态）。
ATTACK_STATES: tuple[str, ...] = ("normal", "spoofing", "jamming")

#: 允许出现的 ``state`` 取值。
#:
#: 注意：开发文档中该字段存在同名两义——§14.2 的输入示例写作 ``"state": "spoofing"``
#: （指攻击类型），而 §13.3 用同一族名字描述事件生命周期状态。为不擅自改写文档语义，
#: 此处同时接受两种取值，并由事件层在写出时统一口径（§13.1 事件定义）。
ALLOWED_STATES: tuple[str, ...] = tuple(
    dict.fromkeys(EVENT_LIFECYCLE_STATES + ATTACK_STATES)
)


@dataclass(slots=True)
class ValidationReport:
    """事件校验报告（不修改输入，只报告问题）。"""

    ok: bool
    missing_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    unknown_key_features: tuple[str, ...] = ()
    invalid_state: str | None = None
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典，便于写入实验记录与日志。"""
        return {
            "ok": self.ok,
            "missing_fields": list(self.missing_fields),
            "unknown_fields": list(self.unknown_fields),
            "unknown_key_features": list(self.unknown_key_features),
            "invalid_state": self.invalid_state,
            "messages": list(self.messages),
        }


def validate_event(event: Mapping[str, Any]) -> ValidationReport:
    """校验事件是否符合 §14.2 的输入契约。

    Args:
        event: 待校验事件字典。

    Returns:
        ValidationReport；``ok=False`` 时由调用方决定是否拒绝生成摘要。

    Note:
        本函数只读：不会增删输入中的任何字段（§14.4 C4 保留原始日志）。
    """
    messages: list[str] = []
    missing = tuple(name for name in REQUIRED_FIELDS if name not in event)
    unknown = tuple(sorted(set(event) - set(ALLOWED_FIELDS)))

    key_features = event.get("key_features") or {}
    unknown_features = tuple(sorted(set(key_features) - set(ALLOWED_KEY_FEATURES))) if isinstance(key_features, Mapping) else ()

    state = event.get("state")
    invalid_state = None
    if isinstance(state, str) and state.lower() not in ALLOWED_STATES:
        invalid_state = state
        messages.append(f"未知事件状态 {state!r}，允许取值：{ALLOWED_STATES}")

    if missing:
        messages.append(f"缺少必需字段：{list(missing)}（§14.4 C3 应显式说明未提供）")
    if unknown:
        messages.append(f"出现白名单外字段：{list(unknown)}（§14.4 C1 只读结构化结果）")
    if unknown_features:
        messages.append(f"key_features 含未定义子字段：{list(unknown_features)}")

    return ValidationReport(
        ok=not (missing or unknown or unknown_features or invalid_state),
        missing_fields=missing,
        unknown_fields=unknown,
        unknown_key_features=unknown_features,
        invalid_state=invalid_state,
        messages=messages,
    )


def summarize_fields(event: Mapping[str, Any]) -> Sequence[str]:
    """返回事件中实际提供的字段名（排序后），用于归档与对比 §18.8 的字段完整性。

    Args:
        event: 事件字典。

    Returns:
        字段名序列。
    """
    return tuple(sorted(event))
