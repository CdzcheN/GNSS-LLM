"""M6 多策略结果融合：把多个检测器输出统一编码并形成结构化判断。

对应开发文档
    §12 章（12.1 结果标准化、12.2 基础融合、12.3 融合规则、12.4 冲突处理）、
    §8.3 多指标冲突场景、§17 评价指标、§18.4 Multi-Strategy Fusion 消融。

职责
    1. 统一编码各检测器输出：R_k = (y_k, p_k, e_k, t_k, q_k, c_k)（§12.1）；
    2. 实现置信度加权融合 S_t = Σ w_k · s_{k,t}（§12.2）；
    3. 识别检测器冲突并给出显式的冲突标记（§12.4），供智能体决定是否追加验证。

不做（边界）
    - 不修改任何单检测器结果（§2.4）；
    - 不再调用检测器（追加调用由 executor/policy 负责，§9.4）；
    - 不做事件级状态机（属 M7，§13）。

输入 / 输出
    输入：DetectionResult 列表 + 可选权重
    输出：FusedResult（含类别、置信度、各检测器贡献与冲突标记）

关键约束
    - 融合必须可解释：输出每个检测器的贡献值，禁止只给最终标签（§2.2）；
    - 权重来源必须显式（环境适用性 / 历史性能 / 数据质量 / 置信度 / 开销，§12.2），
      默认使用“数据质量 × 可靠度”，调用方可完全覆盖；
    - 初期融合顺序遵循 §12.3：规则协调 → 置信度加权 → 时序稳定性 → 事件级融合，
      本模块实现前两步，时序与事件级分别由 fusion 下游与 event 层完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.detectors.base import AttackType, DetectionResult, DetectorStatus

#: 融合步骤顺序（§12.3）。前两步在本模块实现，后两步由下游负责。
FUSION_ORDER: tuple[str, ...] = (
    "rule_coordination",
    "confidence_weighting",
    "temporal_stability",
    "event_level",
)

#: 冲突判据默认参数（§12.4 示例：Jamming 0.82 vs Normal 0.71 → 需追加验证）。
DEFAULT_CONFLICT_MARGIN: float = 0.2
DEFAULT_MIN_CONFIDENCE: float = 0.5


@dataclass(slots=True)
class StandardizedResult:
    """标准化后的单检测器结果 R_k（§12.1）。

    Attributes:
        detector_id: 检测器标识。
        attack_type: 检测类别 y_k。
        confidence: 置信度 p_k。
        evidence: 检测证据 e_k。
        latency_ms: 检测时延 t_k。
        data_quality: 数据质量 q_k。
        cost: 计算成本 c_k。
    """

    detector_id: str
    attack_type: AttackType
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    data_quality: float | None = None
    cost: float | None = None
    status: DetectorStatus = DetectorStatus.OK

    @classmethod
    def from_result(cls, result: DetectionResult) -> "StandardizedResult":
        """由 DetectionResult 构造标准化结果（§12.1）。

        计算成本 c_k 无法从检测结果本身推断，需要时由调用方显式填充。
        """
        return cls(
            detector_id=result.detector_id,
            attack_type=result.attack_type,
            confidence=float(result.confidence),
            evidence=dict(result.evidence),
            latency_ms=result.latency_ms,
            data_quality=result.data_quality,
            status=result.status,
        )


@dataclass(slots=True)
class FusedResult:
    """融合结果（§12.2 输出 + §12.4 冲突标记）。

    Attributes:
        attack_type: 融合后的类别。
        confidence: 融合置信度，取值 [0, 1]。
        score_by_class: 各类别的加权得分，用于解释判定来源。
        contributions: 各检测器对最终类别的贡献（可解释性，§2.2）。
        conflict: 是否存在冲突（§12.4）。
        conflicting_detectors: 与最终结论不一致的检测器。
        used_detectors: 参与融合的检测器 id。
    """

    attack_type: AttackType
    confidence: float
    score_by_class: Mapping[str, float] = field(default_factory=dict)
    contributions: Mapping[str, float] = field(default_factory=dict)
    conflict: bool = False
    conflicting_detectors: tuple[str, ...] = ()
    used_detectors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化字典，供事件记录（§13.1）使用。"""
        return {
            "attack_type": int(self.attack_type),
            "attack_name": self.attack_type.name,
            "confidence": float(self.confidence),
            "score_by_class": dict(self.score_by_class),
            "contributions": dict(self.contributions),
            "conflict": self.conflict,
            "conflicting_detectors": list(self.conflicting_detectors),
            "used_detectors": list(self.used_detectors),
        }


def default_weight(result: StandardizedResult) -> float:
    """默认权重：数据质量 × 可靠度占位（§12.2 权重可考虑的因素之一）。

    Args:
        result: 标准化检测结果。

    Returns:
        非负权重；数据质量缺失时取 1.0。
    """
    quality = 1.0 if result.data_quality is None else float(result.data_quality)
    return max(0.0, quality)


def fuse(
    results: Sequence[DetectionResult] | Sequence[StandardizedResult],
    weights: Mapping[str, float] | None = None,
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> FusedResult:
    """按 §12.2 执行置信度加权融合，并给出 §12.4 冲突标记。

    Args:
        results: 各检测器结果（DetectionResult 或 StandardizedResult）。
        weights: 检测器 id → 权重；未给出的使用 ``default_weight``。
        conflict_margin: 最高与次高（不同类别）置信度差小于该值即判为冲突。
        min_confidence: 判定冲突所需的最低置信度门槛。

    Returns:
        FusedResult。

    Raises:
        ValueError: 结果列表为空，或权重为负。
    """
    if not results:
        raise ValueError("融合输入不能为空（§12.1 需要至少一个检测器结果）")

    standardized = [
        item if isinstance(item, StandardizedResult) else StandardizedResult.from_result(item)
        for item in results
    ]
    # 仅 OK 状态的检测器参与加权；ERROR / SKIPPED 仍计入 used_detectors 以便排查（§2.3）
    informative = [item for item in standardized if item.status is DetectorStatus.OK]
    if not informative:
        return FusedResult(
            attack_type=AttackType.NORMAL,
            confidence=0.0,
            score_by_class={attack.name: 0.0 for attack in AttackType},
            conflict=False,
            used_detectors=tuple(item.detector_id for item in standardized),
        )

    weight_by_id: dict[str, float] = {}
    for item in informative:
        weight = weights.get(item.detector_id) if weights else None
        weight = default_weight(item) if weight is None else float(weight)
        if weight < 0:
            raise ValueError(f"权重不能为负：{item.detector_id}={weight}")
        weight_by_id[item.detector_id] = weight

    score_by_class: dict[str, float] = {attack.name: 0.0 for attack in AttackType}
    for item in informative:
        if item.confidence <= 0.0:
            continue  # 零置信度不贡献得分
        score_by_class[item.attack_type.name] += weight_by_id[item.detector_id] * item.confidence

    best_class_name = max(score_by_class, key=lambda name: score_by_class[name])
    best_class = AttackType[best_class_name]
    total_score = sum(score_by_class.values())
    confidence = (score_by_class[best_class_name] / total_score) if total_score > 0 else 0.0

    contributions = {
        item.detector_id: weight_by_id[item.detector_id] * item.confidence
        for item in informative
        if item.attack_type is best_class
    }
    conflicting = tuple(
        item.detector_id
        for item in informative
        if item.attack_type is not best_class and item.confidence >= min_confidence
    )

    # §12.4：类别不一致且置信度接近 → 判为冲突，交由智能体追加互补检测
    conflict = False
    if conflicting:
        top_two = sorted(
            (score for name, score in score_by_class.items() if name == best_class_name or score > 0),
            reverse=True,
        )
        if len(top_two) >= 2:
            gap = (top_two[0] - top_two[1]) / total_score if total_score > 0 else 0.0
            conflict = gap < conflict_margin

    return FusedResult(
        attack_type=best_class,
        confidence=min(1.0, confidence),
        score_by_class=score_by_class,
        contributions=contributions,
        conflict=conflict,
        conflicting_detectors=conflicting,
        used_detectors=tuple(item.detector_id for item in standardized),
    )


def detect_conflict(
    results: Sequence[DetectionResult],
    margin: float = DEFAULT_CONFLICT_MARGIN,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> bool:
    """独立判断是否存在检测器冲突（§12.4）。

    Args:
        results: 检测结果序列。
        margin: 置信度接近阈值。
        min_confidence: 参与判定的最低置信度。

    Returns:
        存在冲突返回 True。

    Note:
        与 ``fuse()`` 使用同一判据，便于在§9.4 的“是否仍然不确定？”分支中单独调用。
    """
    credible = [
        item for item in results
        if item.status is DetectorStatus.OK and item.confidence >= min_confidence
    ]
    classes = {item.attack_type for item in credible}
    if len(classes) < 2:
        return False
    ordered = sorted((item.confidence for item in credible), reverse=True)
    return (ordered[0] - ordered[1]) < margin
