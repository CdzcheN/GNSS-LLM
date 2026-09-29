"""无监督与半监督辅助路线：正常状态建模与两阶段预训练。

对应开发文档
    §11.1 无监督正常状态建模（Mask-aware LSTM AutoEncoder；输入正常多变量时间窗口，
    输出重建结果与异常分数；用途：正常状态表示学习、少样本辅助、异常程度估计、
    为复杂场景提供额外证据）、
    §11.2 半监督预训练（12–15 正常数据 → 无监督预训练 → 1221 有标签攻击数据 → 监督微调）、
    §11.3 定位（辅助研究路线，不替代「三态检测 + 动态策略选择」主任务）。

职责
    1. 描述自编码器配置（含 mask-aware 开关，对应 §5.4 Q2 的 miss_mask）；
    2. 定义两阶段流程（预训练 + 微调）与各阶段的数据来源；
    3. 定义异常分数接口（重建误差），供 M3 上下文与融合层作为额外证据使用。

不做（边界）
    - 不替代主检测任务（§11.3），其输出只能作为**附加证据**参与融合（§12）；
    - 不实现网络结构与训练循环（需 torch，属后续实现）；
    - 不使用测试/留出数据参与预训练（§16.3 防泄漏）。

输入 / 输出
    输入：正常多变量时间窗口（12–15 正常数据）
    输出：重建结果与异常分数（供 §12 融合作为弱证据）

关键约束
    - 预训练只能使用正常数据（§11.1、§16.2 的 12–13 训练段）；
    - ``mask_aware`` 必须与 §5.4 Q2 的 miss_mask 口径一致，否则会把缺失误判为异常；
    - 该路线的效果不得与主任务指标混报（§18 消融须单列）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: 预训练阶段的数据来源（§11.2：12–15 正常数据）。
PRETRAIN_DATA_SOURCE: str = "normal_12-15"

#: 微调阶段的数据来源（§11.2：1221 有标签攻击数据）。
FINETUNE_DATA_SOURCE: str = "labeled_1221"

#: 该辅助路线在消融实验中的标识（§11.3、§18）。
ABLATION_KEY: str = "unsupervised_pretraining"


@dataclass(slots=True)
class AutoEncoderConfig:
    """Mask-aware LSTM AutoEncoder 配置（§11.1）。

    Attributes:
        window_s: 输入窗口长度（§6.2）。
        hidden_size: 隐藏维度。
        num_layers: LSTM 层数。
        mask_aware: 是否启用掩码感知（对应 §5.4 Q2 的 miss_mask）。
        latent_size: 潜在表示维度。
    """

    window_s: int = 60
    hidden_size: int = 64
    num_layers: int | None = 2
    mask_aware: bool = True
    latent_size: int | None = None

    def __post_init__(self) -> None:
        """校验配置合法性。"""
        if self.window_s <= 0:
            raise ValueError(f"window_s 必须为正：{self.window_s}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size 必须为正：{self.hidden_size}")
        if self.num_layers is not None and self.num_layers <= 0:
            raise ValueError(f"num_layers 必须为正：{self.num_layers}")


@dataclass(slots=True)
class PretrainPlan:
    """两阶段训练流程（§11.2）。

    Attributes:
        stages: 阶段名序列，默认 ``("unsupervised_pretrain", "supervised_finetune")``。
        pretrain_source: 预训练数据来源。
        finetune_source: 微调数据来源。
        notes: 备注（例如“该路线为辅助研究，需与主任务分开报告”）。
    """

    stages: Sequence[str] = field(
        default_factory=lambda: ("unsupervised_pretrain", "supervised_finetune")
    )
    pretrain_source: str = PRETRAIN_DATA_SOURCE
    finetune_source: str = FINETUNE_DATA_SOURCE
    notes: str = "辅助研究路线，不替代主任务（§11.3）"

    def to_dict(self) -> Mapping[str, Any]:
        """导出为可写入实验记录的映射。"""
        return {
            "stages": list(self.stages),
            "pretrain_source": self.pretrain_source,
            "finetune_source": self.finetune_source,
            "ablation_key": ABLATION_KEY,
            "notes": self.notes,
        }


def reconstruction_error(*args: Any, **kwargs: Any) -> Any:
    """计算重建误差（异常分数）的实现入口（§11.1）。

    Raises:
        NotImplementedError: 需 torch 与已训练的自编码器权重。
    """
    raise NotImplementedError(
        "TODO(§11.1): 需 torch；重建误差需按 miss_mask 屏蔽缺失位置后再统计，避免把缺失当作异常"
    )


def anomaly_score(*args: Any, **kwargs: Any) -> Any:
    """由重建误差导出异常分数（§11.1）。

    Raises:
        NotImplementedError: 依赖 ``reconstruction_error`` 的实现。
    """
    raise NotImplementedError("TODO(§11.1): 异常分数应为重建误差的稳定归一化，并记录归一化基准")


def build_autoencoder(*args: Any, **kwargs: Any) -> Any:
    """构建 Mask-aware LSTM AutoEncoder 的实现入口（§11.1）。

    Raises:
        NotImplementedError: 需 torch。
    """
    raise NotImplementedError("TODO(§11.1): 需 torch；网络须显式接收 miss_mask 输入")
