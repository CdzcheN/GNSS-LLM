"""INT8 量化流程与精度损失校验（§19.2）。

对应开发文档
    §19.2 量化策略（FP32 模型 → 验证集评估 → INT8 量化 → 测试集重新评估 → 比较精度损失；
    目标：INT8 后精度损失 < 1%，若无法满足则如实报告）、§22 风险管理（INT8 性能下降）、
    §17.5 系统级指标（INT8 精度变化）。

职责
    1. 固化 §19.2 的五步流程为可执行的数据结构；
    2. 校验量化前后的精度损失是否满足 < 1% 目标；
    3. 生成量化报告（含“是否达标”的明确结论，避免只报数字）。

不做（边界）
    - 不做量化本身（需 torch / onnxruntime，属后续实现）；
    - 不允许隐藏未达标结果：不达标必须显式记录（§19.2 明确要求如实报告）；
    - 不在测试集上做量化校准（校准集应独立，避免污染评估，§16.3 精神）。

输入 / 输出
    输入：量化前后的精度指标、目标阈值
    输出：QuantizationReport

关键约束
    - 精度损失按“相对下降”计算：``(before - after) / before``；
    - 只有在验证集上完成校准后才允许在测试集上复评（§19.2 的步骤顺序不得颠倒）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

#: §19.2 的默认精度损失目标（1%）。
DEFAULT_ACCURACY_DROP_TARGET: float = 0.01

#: §19.2 规定的量化流程（顺序即步骤顺序，不得颠倒）。
QUANTIZATION_STEPS: Sequence[str] = (
    "fp32_train_and_validate",
    "evaluate_on_validation",
    "quantize_int8",
    "reevaluate_on_test",
    "compare_accuracy_drop",
)


@dataclass(slots=True)
class QuantizationReport:
    """量化结果报告（§19.2、§17.5）。

    Attributes:
        metric_name: 用于比较的指标名（如 ``macro_f1``）。
        before: 量化前指标值（FP32）。
        after: 量化后指标值（INT8）。
        relative_drop: 相对精度损失。
        target: 允许的相对损失阈值。
        within_target: 是否满足目标（不满足也必须如实记录）。
        size_before_mb: 量化前模型体积（MB）。
        size_after_mb: 量化后模型体积（MB）。
        notes: 备注（例如量化方法、校准样本数）。
    """

    metric_name: str
    before: float
    after: float
    relative_drop: float
    target: float = DEFAULT_ACCURACY_DROP_TARGET
    within_target: bool = False
    size_before_mb: float | None = None
    size_after_mb: float | None = None
    notes: str = ""

    @property
    def compression_ratio(self) -> float | None:
        """体积压缩比（量化前 / 量化后）；缺少体积数据时返回 ``None``。"""
        if not self.size_before_mb or not self.size_after_mb:
            return None
        return self.size_before_mb / self.size_after_mb

    def to_dict(self) -> Mapping[str, Any]:
        """转为可写入实验记录的映射（§20.3）。"""
        payload = asdict(self)
        payload["compression_ratio"] = self.compression_ratio
        return payload


def relative_drop(before: float, after: float) -> float:
    """计算相对精度损失（§19.2）。

    Args:
        before: 量化前指标值。
        after: 量化后指标值。

    Returns:
        相对下降比例；``before`` 为 0 时返回 0.0（避免除零）。

    Note:
        指标越高越好（如 accuracy / macro_f1）；若使用损失类指标，
        应自行取负后再调用，或改用绝对差比较。
    """
    if before == 0:
        return 0.0
    return (before - after) / before


def check_accuracy_drop(
    before: float,
    after: float,
    target: float = DEFAULT_ACCURACY_DROP_TARGET,
) -> bool:
    """校验精度损失是否满足目标（§19.2）。

    Args:
        before: 量化前指标值。
        after: 量化后指标值。
        target: 允许的相对损失上限（默认 1%）。

    Returns:
        满足返回 True。

    Raises:
        ValueError: ``target`` 为负。
    """
    if target < 0:
        raise ValueError(f"target 不能为负：{target}")
    return relative_drop(before, after) <= target


def build_report(
    metric_name: str,
    before: float,
    after: float,
    target: float = DEFAULT_ACCURACY_DROP_TARGET,
    size_before_mb: float | None = None,
    size_after_mb: float | None = None,
    notes: str = "",
) -> QuantizationReport:
    """构造量化报告（§19.2）。

    Args:
        metric_name: 指标名。
        before: 量化前指标值。
        after: 量化后指标值。
        target: 允许的相对损失上限。
        size_before_mb: 量化前体积（MB）。
        size_after_mb: 量化后体积（MB）。
        notes: 备注。

    Returns:
        QuantizationReport（``within_target`` 给出明确结论；不达标时 notes 会标注需如实报告）。
    """
    drop = relative_drop(before, after)
    within = drop <= target
    if not within and "如实报告" not in notes:
        notes = (notes + " ").strip()
        notes = f"{notes}（未达 §19.2 的 < {target:.0%} 目标，须如实报告）".strip()

    return QuantizationReport(
        metric_name=metric_name,
        before=before,
        after=after,
        relative_drop=drop,
        target=target,
        within_target=within,
        size_before_mb=size_before_mb,
        size_after_mb=size_after_mb,
        notes=notes,
    )


def quantize_dynamic(
    model: Any,
    quantize_linear: bool = True,
    inplace: bool = False,
) -> Any:
    """执行 INT8 动态量化（§19.2 第 3 步）。

    Args:
        model: 待量化的 ``torch.nn.Module``（应已训练完成并处于 eval 模式）。
        quantize_linear: 是否量化 ``nn.Linear`` 层。
        inplace: 是否原地修改模型。

    Returns:
        量化后的模型。

    Raises:
        ImportError: 未安装 PyTorch。
        NotImplementedError: 未提供可量化层时的提示（由 torch 抛出）。

    Note:
        量化后必须回到测试集复评并记录精度损失（§19.2 第 4–5 步），
        不达标时按 §19.2 如实报告，不得隐去。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "量化需要 PyTorch，请执行 `pip install -r requirements.txt`"
        ) from exc

    targets = {torch.nn.Linear} if quantize_linear else {torch.nn.LSTM}
    return torch.quantization.quantize_dynamic(model, targets, inplace=inplace)
