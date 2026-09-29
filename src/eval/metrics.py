"""评价指标：逐秒 / 类别 / 事件 / 策略 / 系统五层（§17）。

对应开发文档
    §17.1 逐秒级指标、§17.2 类别级指标（Spoofing / Jamming 的 P/R/F1，Jamming 必须单列）、
    §17.3 事件级指标、§17.4 策略层指标、§17.5 系统级指标、§2.6 结果优先于指标、
    §16.3 防泄漏（指标只能在同一划分内计算）。

职责
    1. 作为五层指标键的**唯一来源**（``scripts/evaluate.py`` 与 ``config.yaml`` 均引用此处）；
    2. 提供不依赖第三方库的逐秒与类别级指标实现（混淆矩阵 → P/R/F1 → Macro-F1）；
    3. 提供事件级指标实现（基于事件时间区间的匹配）。

不做（边界）
    - 不做显著性检验与统计推断（属后续实验分析）；
    - 不用单一 Accuracy / 单一 F1 代表整体效果（§2.6）；
    - 不计算系统级与策略层的原始测量（分别由 ``src/deploy`` 与编排轨迹提供）。

输入 / 输出
    输入：真实/预测标签序列，或预测与真实事件区间列表
    输出：指标映射

关键约束
    - 类别级指标必须**按类别单独给出**，不允许只报 Macro 值（§17.2 特别要求单列 Jamming）；
    - 未在 ``labels`` 中出现的标签视为输入错误（显式报错，避免静默丢样本）；
    - 事件匹配采用确定性规则（时间重叠即命中），保证结果可复算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.detectors.base import AttackType

#: 逐秒级指标键（§17.1）。
SECOND_LEVEL_METRICS: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "macro_f1", "auc_roc", "auc_pr", "fpr",
)

#: 类别级指标键（§17.2：Spoofing 与 Jamming 各自的 Precision / Recall / F1）。
CLASS_LEVEL_METRICS: tuple[str, ...] = ("spoofing_p_r_f1", "jamming_p_r_f1")

#: 事件级指标键（§17.3）。
EVENT_LEVEL_METRICS: tuple[str, ...] = (
    "event_detection_rate", "event_recall", "detection_delay",
    "false_alarm_events", "missed_events", "event_level_precision",
)

#: 策略层指标键（§17.4）。
STRATEGY_LEVEL_METRICS: tuple[str, ...] = (
    "avg_detector_calls", "deep_model_call_rate", "avg_decision_time",
    "avg_detector_time", "conflict_rate", "recheck_trigger_rate",
)

#: 系统级指标键（§17.5）。
SYSTEM_LEVEL_METRICS: tuple[str, ...] = (
    "end_to_end_latency", "throughput", "memory", "parameters",
    "int8_delta", "llm_summary_latency", "alert_availability",
)

#: 默认三态标签集合（§1.3）。
DEFAULT_LABELS: tuple[int, ...] = tuple(int(attack) for attack in AttackType)

#: 类别名 → 指标键前缀（§17.2）。
CLASS_METRIC_KEYS: Mapping[AttackType, str] = {
    AttackType.SPOOFING: "spoofing",
    AttackType.JAMMING: "jamming",
}


def confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict[int, dict[int, int]]:
    """构建混淆矩阵（行 = 真实，列 = 预测）。

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        labels: 参与统计的标签集合。

    Returns:
        ``{真实标签: {预测标签: 计数}}``。

    Raises:
        ValueError: 序列长度不一致，或出现 ``labels`` 之外的标签。
    """
    true_list = [_normalize(label, labels) for label in y_true]
    pred_list = [_normalize(label, labels) for label in y_pred]
    if len(true_list) != len(pred_list):
        raise ValueError(f"真实与预测序列长度不一致：{len(true_list)} vs {len(pred_list)}")

    matrix = {label: {other: 0 for other in labels} for label in labels}
    for truth, pred in zip(true_list, pred_list):
        matrix[truth][pred] += 1
    return matrix


def _normalize(label: Any, labels: Sequence[int]) -> int:
    """把标签规整为 int 并校验是否在允许集合内。"""
    try:
        value = int(label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标签无法转换为整数：{label!r}") from exc
    if value not in labels:
        raise ValueError(f"出现未定义的标签 {value}，允许集合：{tuple(labels)}（§1.3）")
    return value


def per_class_metrics(matrix: Mapping[int, Mapping[int, int]], label: int) -> Mapping[str, float]:
    """计算单个类别的 Precision / Recall / F1 与样本数。

    Args:
        matrix: 混淆矩阵。
        label: 目标类别。

    Returns:
        含 ``precision`` / ``recall`` / ``f1`` / ``support`` 的映射（分母为 0 时记 0.0）。
    """
    true_positive = float(matrix[label][label])
    predicted = float(sum(matrix[row][label] for row in matrix))
    actual = float(sum(matrix[label].values()))

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual}


def accuracy(matrix: Mapping[int, Mapping[int, int]]) -> float:
    """逐秒 Accuracy（§17.1）。

    Args:
        matrix: 混淆矩阵。

    Returns:
        Accuracy；总样本为 0 时返回 0.0。
    """
    total = sum(sum(row.values()) for row in matrix.values())
    if not total:
        return 0.0
    correct = sum(matrix[label][label] for label in matrix)
    return correct / total


def macro_f1(
    matrix: Mapping[int, Mapping[int, int]],
    labels: Sequence[int] = DEFAULT_LABELS,
) -> float:
    """Macro-F1（§17.1；§1.4 G2 以 0.90 为目标值，但不是唯一验收条件）。

    Args:
        matrix: 混淆矩阵。
        labels: 参与平均的类别。

    Returns:
        各类别 F1 的算术平均。
    """
    scores = [per_class_metrics(matrix, label)["f1"] for label in labels]
    return sum(scores) / len(scores) if scores else 0.0


def classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[int] = DEFAULT_LABELS,
) -> Mapping[str, Any]:
    """生成逐秒与类别级指标报告（§17.1、§17.2）。

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        labels: 参与统计的标签集合。

    Returns:
        含 ``accuracy`` / ``macro_f1`` / ``per_class`` / ``spoofing`` / ``jamming`` 的映射，
        其中 ``per_class`` 以类别名小写为键（与 §17.2 的报告口径一致）。
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    per_class = {
        AttackType(label).name.lower(): dict(per_class_metrics(matrix, label)) for label in labels
    }
    report: dict[str, Any] = {
        "accuracy": accuracy(matrix),
        "macro_f1": macro_f1(matrix, labels),
        "per_class": per_class,
        "support_total": sum(sum(row.values()) for row in matrix.values()),
        "metrics_keys": list(SECOND_LEVEL_METRICS),
    }
    for attack, prefix in CLASS_METRIC_KEYS.items():
        label = int(attack)
        report[prefix] = per_class.get(attack.name.lower(), {})
        report[f"{prefix}_p_r_f1"] = per_class.get(attack.name.lower(), {})
    return report


@dataclass(frozen=True, slots=True)
class Interval:
    """事件时间区间（秒），用于事件级匹配（§17.3、§13.1）。"""

    start: float
    end: float
    label: str = ""

    def overlaps(self, other: "Interval") -> bool:
        """判断两个区间是否时间重叠（端点相接不算重叠）。"""
        return self.start < other.end and other.start < self.end


def match_events(
    predicted: Sequence[Interval],
    truth: Sequence[Interval],
) -> Mapping[str, Any]:
    """按时间重叠匹配预测事件与真实事件（§17.3）。

    Args:
        predicted: 预测事件区间。
        truth: 真实事件区间。

    Returns:
        含 ``matched``（成对区间）、``missed``（漏检的真实事件）、
        ``false_alarms``（误报的预测事件）的映射。

    Note:
        采用“时间重叠即命中”的确定性规则；一个预测最多匹配一个真实事件（贪心顺序匹配），
        保证结果可复算。更严格的重叠比例判据可在实验协议中另行约定。
    """
    used_truth: set[int] = set()
    matched: list[tuple[Interval, Interval]] = []
    false_alarms: list[Interval] = []

    for prediction in predicted:
        for index, target in enumerate(truth):
            if index in used_truth:
                continue
            if prediction.overlaps(target):
                used_truth.add(index)
                matched.append((prediction, target))
                break
        else:
            false_alarms.append(prediction)

    missed = [target for index, target in enumerate(truth) if index not in used_truth]
    return {"matched": matched, "missed": missed, "false_alarms": false_alarms}


def event_level_metrics(
    predicted: Sequence[Interval],
    truth: Sequence[Interval],
) -> Mapping[str, float | int]:
    """计算事件级指标（§17.3）。

    Args:
        predicted: 预测事件区间。
        truth: 真实事件区间。

    Returns:
        含 ``event_detection_rate`` / ``event_recall`` / ``detection_delay`` /
        ``false_alarm_events`` / ``missed_events`` / ``event_level_precision`` 的映射。

    Note:
        文档未给出 ``event_detection_rate`` 与 ``event_recall`` 的区分口径，
        此处按“检出率 = 被至少匹配一次的预测事件比例”实现，二者分别对应
        真实侧与预测侧的召回视角，避免重复定义。
    """
    result = match_events(predicted, truth)
    matched = result["matched"]
    delays = [prediction.start - target.start for prediction, target in matched]

    return {
        "event_detection_rate": (len(matched) / len(predicted)) if predicted else 0.0,
        "event_recall": (len(matched) / len(truth)) if truth else 0.0,
        "detection_delay": (sum(delays) / len(delays)) if delays else 0.0,
        "false_alarm_events": len(result["false_alarms"]),
        "missed_events": len(result["missed"]),
        "event_level_precision": (len(matched) / len(predicted)) if predicted else 0.0,
    }
