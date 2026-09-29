"""训练损失与类别不平衡处理（加权 CE 与 Focal Loss）。

对应开发文档
    §10.5 类别不平衡（类别加权 CE：Normal 1.0 / Spoofing ≈ 3.9 / Jamming ≈ 70；
    Focal Loss γ = 2；并支持训练集重采样、时序增强、少数类事件级采样）、
    §5.4 Q5 类别不平衡（Jamming 仅 481 s）、§6.4 / §16.3（统计量只能来自训练集）。

职责
    1. 由**训练集**类别频数推导加权 CE 的类别权重，并与 §10.5 的基线值核对；
    2. 给出 Focal Loss 的参数配置与实现入口；
    3. 记录重采样 / 时序增强 / 少数类事件级采样等策略开关（§10.5）。

不做（边界）
    - 不使用测试集或全量数据统计类别频数（§16.3 防泄漏，§6.4 只允许训练集统计量）；
    - 不改变 §10.5 给出的基线权重，只提供推导、核对与替换能力；
    - 不实现数据增强本身（属数据集构建环节）。

输入 / 输出
    输入：训练集类别计数 ``{类别: 样本数}``
    输出：类别权重映射与损失配置（LossSpec）

权重口径
    采用与 scikit-learn ``class_weight="balanced"`` 相同的口径：``w_c = N / (K · n_c)``，
    再以样本最多的类别为参照归一化（Normal = 1.0）。
    验证：以 §5.2 的 1221 计数（Normal 33847 / Spoofing 8604 / Jamming 481）代入，
    得到 Normal 1.0、Spoofing ≈ 3.93、Jamming ≈ 70.4，与 §10.5 的基线值一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.detectors.base import AttackType
from src.detectors.deep_temporal import (
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_FOCAL_GAMMA,
)

#: §5.2 给出的 1221 三态计数，用于核对类别权重口径。
BASELINE_CLASS_COUNTS: Mapping[int, int] = {
    int(AttackType.NORMAL): 33847,
    int(AttackType.SPOOFING): 8604,
    int(AttackType.JAMMING): 481,
}

#: §10.5 支持的训练损失。
LOSS_NAMES: tuple[str, ...] = ("ce", "weighted_ce", "focal")

#: §10.5 支持的类别不平衡缓解策略。
IMBALANCE_STRATEGIES: tuple[str, ...] = (
    "resample",
    "temporal_augment",
    "minority_event_sampling",
)


def balanced_class_weights(
    counts: Mapping[int, float],
    normalize_reference: int | None = None,
) -> dict[int, float]:
    """按训练集类别频数推导类别权重。

    Args:
        counts: 类别 → 训练集样本数（必须为正）。
        normalize_reference: 归一化参照类（其权重为 1.0）；
            ``None`` 表示取样本数最多的类别。

    Returns:
        类别 → 权重映射。

    Raises:
        ValueError: ``counts`` 为空，或存在非正计数。

    Note:
        公式 ``w_c = N / (K · n_c)`` 后除以参照类的原始权重，
        因此参照类权重恒为 1.0。文档未给出公式，此处采用业界通用且可复算的口径，
        并可用 ``verify_baseline_weights`` 与 §10.5 的基线值对账。
    """
    if not counts:
        raise ValueError("counts 不能为空（需要训练集类别频数）")
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"类别样本数必须为正：{dict(counts)}")

    total = float(sum(counts.values()))
    classes = len(counts)
    raw = {label: total / (classes * float(value)) for label, value in counts.items()}

    reference = normalize_reference
    if reference is None:
        reference = max(counts, key=lambda label: counts[label])
    if reference not in raw:
        raise ValueError(f"参照类 {reference!r} 不在 counts 中")

    base = raw[reference]
    return {label: weight / base for label, weight in raw.items()}


def verify_baseline_weights(
    tolerance: float = 0.05,
    counts: Mapping[int, int] = BASELINE_CLASS_COUNTS,
    baseline: Mapping[int, float] = DEFAULT_CLASS_WEIGHTS,
) -> Mapping[str, Any]:
    """核对推导权重与 §10.5 基线权重是否一致。

    Args:
        tolerance: 允许的相对误差。
        counts: 用于推导的类别计数（默认 §5.2 的 1221 计数）。
        baseline: 文档基线权重（默认 §10.5）。

    Returns:
        含 ``ok``、``derived``、``baseline``、``max_relative_error`` 的映射。
    """
    derived = balanced_class_weights(counts)
    errors = {
        label: abs(derived[label] - float(baseline[label])) / float(baseline[label])
        for label in derived
        if label in baseline
    }
    max_error = max(errors.values()) if errors else 0.0
    return {
        "ok": max_error <= tolerance,
        "derived": derived,
        "baseline": dict(baseline),
        "max_relative_error": max_error,
    }


@dataclass(slots=True)
class LossSpec:
    """训练损失配置（§10.5）。

    Attributes:
        name: 损失类型，取值见 ``LOSS_NAMES``。
        class_weights: 类别权重（加权 CE 使用；``None`` 表示不加权）。
        focal_gamma: Focal Loss 的 γ（§10.5 基线为 2）。
        strategies: 启用的类别不平衡缓解策略，取值见 ``IMBALANCE_STRATEGIES``。
    """

    name: str = "weighted_ce"
    class_weights: Mapping[int, float] | None = field(default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS))
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    strategies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验损失名与策略名合法性。"""
        if self.name not in LOSS_NAMES:
            raise ValueError(f"未知损失 {self.name!r}，允许：{LOSS_NAMES}")
        unknown = [item for item in self.strategies if item not in IMBALANCE_STRATEGIES]
        if unknown:
            raise ValueError(f"未知不平衡策略 {unknown}，允许：{IMBALANCE_STRATEGIES}")
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma 不能为负：{self.focal_gamma}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LossSpec":
        """由 config.yaml 的 ``class_imbalance`` 段构造。

        Args:
            config: 含 ``class_weight`` / ``focal_loss_gamma`` / ``support`` 的映射。

        Returns:
            LossSpec。
        """
        label_map = {
            "Normal": int(AttackType.NORMAL),
            "Spoofing": int(AttackType.SPOOFING),
            "Jamming": int(AttackType.JAMMING),
        }
        weights = config.get("class_weight") or {}
        mapped = {label_map[name]: float(value) for name, value in weights.items() if name in label_map}
        return cls(
            class_weights=mapped or None,
            focal_gamma=float(config.get("focal_loss_gamma", DEFAULT_FOCAL_GAMMA)),
            strategies=tuple(config.get("support", ()) or ()),
        )


def weighted_cross_entropy(*args: Any, **kwargs: Any) -> Any:
    """加权交叉熵实现入口（§10.5）。

    Raises:
        NotImplementedError: 需要 PyTorch；实现时应在模型训练阶段惰性导入 torch
            并直接使用 ``torch.nn.CrossEntropyLoss(weight=...)``。
    """
    raise NotImplementedError("TODO(§10.5): 需 torch；请使用 torch.nn.CrossEntropyLoss(weight=类别权重)")


def focal_loss(*args: Any, **kwargs: Any) -> Any:
    """Focal Loss 实现入口（§10.5，γ = 2）。

    Raises:
        NotImplementedError: 需 torch；实现须保证数值稳定（log_softmax 形式）。
    """
    raise NotImplementedError("TODO(§10.5): 需 torch；Focal Loss 应在 log_softmax 上实现以保证数值稳定")
