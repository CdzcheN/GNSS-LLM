"""M8 提示词模板与版本管理：约束大模型只做“信息表达”，不做任何判定。

对应开发文档
    §14.2 LLM 输入（只允许结构化字段）、§14.3 LLM 输出（事件摘要 / 检测过程摘要 / 监测报告）、
    §14.4 约束 C1–C5、§14.5 工程价值、§2.4 大模型不参与核心判定。

职责
    1. 提供固定模板：把结构化事件转成提示词；
    2. 通过版本号（``PROMPT_VERSION``）支持提示词可复现（§14.4 C5）；
    3. 在模板中显式声明禁止项，降低幻觉风险（§14.4 C1–C3）。

不做（边界）
    - 不调用任何模型（调用在 summarizer 中）；
    - 不在提示词中引入事件以外的信息（不得注入原始 GNSS 数据，§14.4 C1）；
    - 不允许模板引导模型“推测原因”或补充未给出的证据（§14.4 C3）。

输入 / 输出
    输入：结构化事件字典（字段见 §14.2）
    输出：提示词字符串

关键约束
    - 模板必须包含“不得修改标签/置信度/时间、不得补造证据”的明确约束语句（§2.4、§14.4）；
    - 提示词版本变化必须同步更新 ``CHANGELOG`` 与实验记录（§28 文档维护规则）。
"""

from __future__ import annotations

import json
from string import Template
from typing import Any, Mapping

#: 提示词版本（§14.4 C5：必须随输出一并保存）。
PROMPT_VERSION: str = "v1"

#: 系统约束语句。逐条对应 §14.4 的五项约束与 §2.4 的职责边界。
SYSTEM_CONSTRAINTS: str = (
    "你是 GNSS 干扰监测系统的日志总结助手。\n"
    "严格约束：\n"
    "1) 只使用输入 JSON 中出现的字段，不得补充或推测任何未给出的信息；\n"
    "2) 不得修改事件时间、类别、置信度、检测策略等结构化字段；\n"
    "3) 缺少的信息必须显式说明“未提供”，不得写成确定事实；\n"
    "4) 不给出结构化结果之外的结论，不做因果推断；\n"
    "5) 输出为纯文本摘要，不输出 JSON、代码块或建议性指令。"
)

#: 提示词模板。``$event_json`` 由 build_prompt 注入规范化后的结构化事件。
PROMPT_TEMPLATE: Template = Template(
    "$constraints\n\n"
    "以下是结构化检测事件（唯一事实来源）：\n"
    "$event_json\n\n"
    "请按以下三项产出：\n"
    "A. 事件摘要：简洁，只陈述事实；\n"
    "B. 检测过程摘要：说明调用了哪些检测策略、哪一步触发了追加检测、"
    "各检测器给出的证据、最终结构化结论；\n"
    "C. 监测报告：按事件编号 / 时间范围 / 异常类型 / 置信度 / 检测时延 / "
    "主要异常表现 / 调用检测策略 / 系统结构化结论 的固定格式输出。\n"
)


def build_prompt(event: Mapping[str, Any], prompt_version: str = PROMPT_VERSION) -> str:
    """把结构化事件渲染为提示词。

    Args:
        event: 结构化事件字典（字段见 §14.2，例如 event_id / start_time / state /
            confidence / selected_strategies / key_features / data_quality）。
        prompt_version: 提示词版本号，仅用于记录（模板内容由本模块维护）。

    Returns:
        完整提示词字符串。

    Raises:
        ValueError: 事件为空或缺少 ``event_id``（无法溯源，§14.4 C5）。
    """
    if not event:
        raise ValueError("事件不能为空（§14.2 只允许输入结构化信息）")
    if not event.get("event_id"):
        raise ValueError("事件缺少 event_id，无法满足 §14.4 C5 的可复现要求")

    payload = json.dumps(dict(event), ensure_ascii=False, sort_keys=True, indent=2)
    rendered = PROMPT_TEMPLATE.substitute(constraints=SYSTEM_CONSTRAINTS, event_json=payload)
    return f"# prompt_version={prompt_version}\n{rendered}"
