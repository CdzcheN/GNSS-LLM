"""消融与对比实验定义（§18）。

对应开发文档
    §18.1 Context 消融、§18.2 Dynamic Selection 消融、§18.3 Agent 消融、
    §18.4 Multi-Strategy Fusion 消融、§18.5 Deep Detector 消融、§18.6 模态消融、
    §18.7 窗口敏感性、§18.8 LLM 模块验证。

职责
    1. 把 §18 的八个消融项固化为可执行的数据结构（对照组清单与开关位置）；
    2. 作为 ``scripts/evaluate.py --ablation`` 的取值来源，保证命令行与文档一致；
    3. 记录每个消融项对应的**代码开关**，避免“实验做了但改错了地方”。

不做（边界）
    - 不执行实验（由 scripts/evaluate.py 调用具体实现）；
    - 不判定结论（结论由研究者依据指标填写，§20.3）；
    - 不重复定义 §17 的指标键（统一由 ``src/eval/metrics.py`` 提供）。

输入 / 输出
    输入：无（定义性数据）
    输出：AblationSpec 映射与命令行取值
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """单项消融实验的定义（§18）。

    Attributes:
        key: 命令行取值（``--ablation``）。
        section: 对应的开发文档小节。
        arms: 对照组清单（按文档顺序）。
        switch: 该消融在代码中的开关位置说明。
        note: 口径备注。
    """

    key: str
    section: str
    arms: Sequence[str]
    switch: str
    note: str = ""


#: §18 的八个消融项（顺序与文档一致）。
DEFAULT_ABLATIONS: Mapping[str, AblationSpec] = {
    spec.key: spec
    for spec in (
        AblationSpec(
            key="context",
            section="§18.1",
            arms=("no_context", "with_context"),
            switch="src/context/context_encoder.py：ContextEncoder(enabled=False) 即对照组",
            note="验证环境状态表示的贡献",
        ),
        AblationSpec(
            key="dynamic_selection",
            section="§18.2",
            arms=("fixed_policy", "context_rule_policy"),
            switch="src/agent/policy.py：FixedPolicy（Level 0）vs ContextRulePolicy（Level 1）",
            note="验证核心创新（动态策略选择）",
        ),
        AblationSpec(
            key="agent",
            section="§18.3",
            arms=("rule_router", "learning_router", "agent_router"),
            switch="src/agent/policy.py：Policy.level = 1 / 2 / 3",
            note="Level 2 与 Level 3 尚未实现，见文件头部“待实现”",
        ),
        AblationSpec(
            key="fusion",
            section="§18.4",
            arms=("single_detector", "multi_no_fusion", "multi_fusion"),
            switch="src/fusion/result_fusion.py：fuse() 与 detect_conflict()",
            note="对比单检测器 / 多检测器不融合 / 多检测器融合",
        ),
        AblationSpec(
            key="deep_detector",
            section="§18.5",
            arms=("without_deep", "with_deep"),
            switch="src/agent/planner.py：DEFAULT_APPLICABILITY 中是否允许 deep_temporal",
            note="验证深度时序检测在复杂/不确定场景中的作用",
        ),
        AblationSpec(
            key="modality",
            section="§18.6",
            arms=("A", "A+B", "A+C", "A+B+C"),
            switch="src/features/*：按模态选择参与编码的特征组",
            note="验证频谱/AGC（B）与 RAWX（C）对系统的增量贡献",
        ),
        AblationSpec(
            key="window",
            section="§18.7",
            arms=("10", "30", "60", "120"),
            switch="src/train/dataset.py：WindowConfig.window_s（取值集合 WINDOW_CHOICES）",
            note="窗口敏感性实验",
        ),
        AblationSpec(
            key="llm",
            section="§18.8",
            arms=("template_summary", "llm_summary"),
            switch="src/llm/summarizer.py：TemplateSummarizer vs LLMSummarizer",
            note="LLM 不参与算法准确率比较，只评价事实一致性、字段完整性、可读性、生成延迟",
        ),
    )
}

#: 命令行可用的消融取值（与 §18 小节一一对应）。
ABLATION_KEYS: tuple[str, ...] = tuple(DEFAULT_ABLATIONS)


def get_ablation(key: str) -> AblationSpec:
    """按 key 取消融定义。

    Args:
        key: 消融标识（``--ablation`` 的取值）。

    Returns:
        AblationSpec。

    Raises:
        KeyError: 未定义的消融项（消息中列出可用取值）。
    """
    try:
        return DEFAULT_ABLATIONS[key]
    except KeyError as exc:
        raise KeyError(f"未定义的消融项 {key!r}，可用：{ABLATION_KEYS}") from exc


def validate_arms(key: str, arms: Sequence[str]) -> Sequence[str]:
    """校验实验给出的对照组是否属于该消融项（§18）。

    Args:
        key: 消融标识。
        arms: 待校验的对照组清单；空序列表示使用文档定义的全部对照组。

    Returns:
        校验后的对照组元组。

    Raises:
        KeyError: 消融项未定义。
        ValueError: 出现该消融项之外的对照组。
    """
    spec = get_ablation(key)
    if not arms:
        return tuple(spec.arms)
    unknown = [arm for arm in arms if arm not in spec.arms]
    if unknown:
        raise ValueError(f"{spec.section} 未定义的对照组 {unknown}，可用：{tuple(spec.arms)}")
    return tuple(arms)


def summary() -> Sequence[Mapping[str, Any]]:
    """导出全部消融项摘要，便于写入实验记录（§20.3）或打印到控制台。"""
    return [
        {
            "key": spec.key,
            "section": spec.section,
            "arms": list(spec.arms),
            "switch": spec.switch,
        }
        for spec in DEFAULT_ABLATIONS.values()
    ]
