"""M8 日志总结器：只读结构化事件、异步生成摘要（不进入实时检测关键路径）。

对应开发文档
    §14.1 模块定位、§14.2 输入、§14.3 输出、§14.4 约束 C1–C5、§14.5 工程价值、
    §15.2 实时性原则（异步、不阻塞）、§19.4 LLM 部署原则、§2.4 大模型不参与核心判定。

职责
    1. 定义可插拔的摘要器接口 Summarizer；
    2. 提供确定性降级实现 TemplateSummarizer（无 LLM 时仍可产出报告，§14.4 C4）；
    3. 固化调用记录 SummarizationRecord（§14.4 C5：model / prompt 版本、输入事件 id、输出、时间）。

不做（边界）
    - 不读取原始 GNSS 数据、不访问检测器（§14.4 C1 只读结构化结果）；
    - 不修改事件标签、置信度或时间（§2.4）；
    - 不阻塞实时告警链路（§15.2、§19.4）。

输入 / 输出
    输入：结构化事件字典（§14.2，须先通过 schema.validate_event）
    输出：SummarizationRecord（含摘要文本与可复现元数据）

关键约束
    - 未通过 schema 校验的事件必须拒绝，避免把越界内容喂给模型（§14.4 C1）；
    - 每次调用都必须落盘可复现元数据（§14.4 C5、§20.3）；
    - 摘要失败不得影响告警：调用方应在 try/except 中异步调用（§15.2、§19.4）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from src.llm.prompt import PROMPT_VERSION, build_prompt
from src.llm.report import build_report
from src.llm.schema import validate_event


@dataclass(slots=True)
class SummarizationRecord:
    """一次摘要调用的可复现记录（字段对应 §14.4 C5）。"""

    input_event_id: str
    summary_text: str
    model_name: str
    model_version: str
    prompt_version: str = PROMPT_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    report_text: str | None = None
    validation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为可写盘的字典（原始日志保留，§14.4 C4）。"""
        return asdict(self)


class Summarizer(ABC):
    """摘要器接口（§14.1、§14.5）。"""

    #: 提供方名称，用于记录（§14.4 C5）。
    provider: str = "abstract"
    #: 模型名与版本，子类必须给出真实取值以便复现。
    model_name: str = "unknown"
    model_version: str = "unknown"

    @abstractmethod
    def summarize(self, event: Mapping[str, Any]) -> SummarizationRecord:
        """生成事件摘要与报告。

        Args:
            event: 结构化事件字典（§14.2）。

        Returns:
            SummarizationRecord。

        Raises:
            ValueError: 事件未通过 schema 校验。
        """
        raise NotImplementedError


class TemplateSummarizer(Summarizer):
    """确定性降级摘要器：不调用 LLM，仅按 §14.3 模板转述结构化字段。

    用于两类场景：LLM 服务不可用时保证报告可用（§14.4 C4）；作为 LLM 输出的
    字段一致性对照基准（§18.8）。
    """

    provider = "template"
    model_name = "template"
    model_version = "v1"

    def summarize(self, event: Mapping[str, Any]) -> SummarizationRecord:
        """生成模板化摘要与监测报告。

        Args:
            event: 结构化事件字典。

        Returns:
            SummarizationRecord。

        Raises:
            ValueError: 未通过 §14.2 契约校验（不生成任何推测性内容）。
        """
        report = validate_event(event)
        if not report.ok:
            raise ValueError(f"事件未通过 schema 校验：{report.messages}")

        summary = (
            f"事件 {event.get('event_id')}：状态 {event.get('state')}，"
            f"置信度 {event.get('confidence')}，"
            f"调用策略 {list(event.get('selected_strategies') or [])}。"
        )
        return SummarizationRecord(
            input_event_id=str(event.get("event_id")),
            summary_text=summary,
            report_text=build_report(event),
            model_name=self.model_name,
            model_version=self.model_version,
            validation=report.to_dict(),
        )


class LLMSummarizer(Summarizer):
    """大模型摘要器（可插拔，待接入具体服务）。

    注意事项（对应 §14.4 与 §19.4）：
        - 必须在异步线程/任务中调用，禁止阻塞 1 Hz 实时链路；
        - 只传入 ``build_prompt`` 渲染后的文本，不得追加原始 GNSS 数据；
        - 输出需与结构化字段做一致性校验（§18.8），不一致时以结构化结果为准。
    """

    provider = "llm"

    def __init__(self, model_name: str = "", model_version: str = "unknown", client: Any = None) -> None:
        """初始化。

        Args:
            model_name: 模型名（记录用，§14.4 C5）。
            model_version: 模型版本（记录用）。
            client: 已初始化的 SDK 客户端；``None`` 表示尚未配置。
        """
        self.model_name = model_name or "unknown"
        self.model_version = model_version
        self.client = client

    def build_request(self, event: Mapping[str, Any]) -> str:
        """生成待发送的提示词（便于单测与留痕）。"""
        return build_prompt(event)

    def summarize(self, event: Mapping[str, Any]) -> SummarizationRecord:
        """调用大模型生成摘要。

        Args:
            event: 结构化事件字典。

        Returns:
            SummarizationRecord。

        Raises:
            ValueError: 事件未通过 schema 校验。
            NotImplementedError: 尚未接入具体 LLM 提供方（见 §14.5 可插拔设计）。

        对应开发文档：§14.1、§14.4、§15.2、§19.4。
        """
        report = validate_event(event)
        if not report.ok:
            raise ValueError(f"事件未通过 schema 校验：{report.messages}")
        raise NotImplementedError(
            "TODO(§14.5): 接入具体 LLM 提供方；调用需异步执行且不得阻塞实时告警（§19.4）"
        )
