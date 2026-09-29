"""M8 监测报告生成：按 §14.3 的固定格式产出面向人员的报告文本。

对应开发文档
    §14.3 输出 3（监测报告格式）、§14.5 模块的工程价值、§18.8 LLM 模块验证。

职责
    1. 提供与 §14.3 完全一致的报告模板（事件编号 / 时间范围 / 异常类型 / 置信度 /
       检测时延 / 主要异常表现 / 调用检测策略 / 系统结构化结论）；
    2. 在无 LLM 可用时提供确定性的降级输出，保证 §14.4 C4 的日志可用性；
    3. 保证报告文本与结构化字段逐项一致（§18.8 的一致性检查对象）。

不做（边界）
    - 不生成未经结构化数据支持的“原因分析”（§14.4 C3）；
    - 不替代原始 JSON/CSV 事件记录（§14.4 C4）；
    - 不改写事件字段（§2.4）。

输入 / 输出
    输入：结构化事件字典（§14.2）与可选证据列表
    输出：报告文本（str）

关键约束
    - 模板结构必须与 §14.3 一致，便于人工与自动核对；
    - 缺失字段必须以“未提供”显式标注，禁止留空或编造（§14.4 C3）；
    - 报告生成必须是确定性函数：同输入同输出（§20.4）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: 缺失值占位符（§14.4 C3：未知内容必须显式说明）。
MISSING: str = "未提供"

#: 报告模板（结构与 §14.3 “输出 3：监测报告”一致）。
REPORT_TEMPLATE: str = (
    "事件编号：{event_id}\n"
    "时间范围：{time_range}\n"
    "异常类型：{state}\n"
    "置信度：{confidence}\n"
    "检测时延：{detection_delay}\n"
    "\n"
    "主要异常表现：\n"
    "{key_features}\n"
    "\n"
    "调用检测策略：\n"
    "{strategies}\n"
    "\n"
    "系统结构化结论：\n"
    "{conclusion}\n"
)


def _number(value: Any, suffix: str = "", digits: int = 2) -> str:
    """把数值格式化为带单位的字符串；缺失时返回占位符。

    Args:
        value: 原始值。
        suffix: 单位后缀，例如 ``" s"``。
        digits: 小数位数。

    Returns:
        格式化字符串，或 ``MISSING``。
    """
    if value is None or value == "":
        return MISSING
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _bullets(items: Sequence[Any]) -> str:
    """把序列渲染为编号列表；空序列返回占位符。"""
    if not items:
        return f"1. {MISSING}"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def build_report(event: Mapping[str, Any], conclusion: str | None = None) -> str:
    """按 §14.3 格式生成监测报告。

    Args:
        event: 结构化事件字典（§14.2）。
        conclusion: 系统结构化结论文本；``None`` 时由事件字段拼装。

    Returns:
        报告文本。

    Note:
        本函数是确定性模板渲染，不依赖 LLM；在 LLM 不可用时作为降级输出，
        也可作为 LLM 输出的“字段一致性”对照基准（§18.8）。
    """
    start = event.get("start_time", MISSING)
    end = event.get("end_time", MISSING)
    time_range = f"{start} ~ {end}" if (start != MISSING or end != MISSING) else MISSING

    features = event.get("key_features")
    if isinstance(features, Mapping) and features:
        feature_lines = [f"{name} = {value}" for name, value in sorted(features.items())]
    else:
        feature_lines = []

    strategies = event.get("selected_strategies") or []

    if conclusion is None:
        conclusion = (
            f"结构化判定为 {event.get('state', MISSING)}，"
            f"置信度 {_number(event.get('confidence'))}，"
            f"数据质量 {_number(event.get('data_quality'))}。"
            "本报告仅转述结构化检测结果，不含额外推断。"
        )

    return REPORT_TEMPLATE.format(
        event_id=event.get("event_id", MISSING),
        time_range=time_range,
        state=event.get("state", MISSING),
        confidence=_number(event.get("confidence")),
        detection_delay=_number(event.get("detection_delay_s"), suffix=" s", digits=0),
        key_features=_bullets(feature_lines),
        strategies=_bullets(list(strategies)),
        conclusion=conclusion,
    )
