"""M2 频谱特征：频谱能量、频谱形态与可选 128 维谱表示（模态 B）。

对应开发文档
    §5.3 模态 B（AGC / 频谱）、§6.3 派生特征（频谱能量变化）、§7.3 信号类特征、
    §8.2 S2 频谱检测、§8.3 策略适用性矩阵（频谱异常明显 → 频谱检测）。

职责
    1. 从 MON-SPAN 频谱数据构造可解释的统计特征（带宽、峰值、带内/带外能量比）；
    2. 提供可选的 128 维谱表示，供深度模型与频谱检测器复用；
    3. 输出“频谱异常明显”这一上下文判据所需的中间量。

不做（边界）
    - 不做射频判决（阈值与结论属频谱检测器 S2，§8.2）；
    - 不做原始 I/Q 全链路处理（明确不在范围内，见 §1.5 暂不包含）；
    - 不输出无法解释的稠密表示作为唯一证据（§2.2）。

输入 / 输出
    输入：MON-SPAN 频谱矩阵（时间 × 频点）
    输出：频谱统计特征表；可选 128 维谱表示矩阵

关键约束
    - 谱表示维度固定为 128（§5.3 模态 B），以便模态消融实验（§18.6）可复现；
    - 谱特征必须按时间先后计算，不得跨窗口混入未来信息（§2.5）；
    - 频谱统计量需与 AGC 联动核对，避免把接收机增益变化误判为干扰。

待实现
    - spectrum_statistics()：峰值/带宽/能量比等统计特征
    - spectrum_embedding()：定长 128 维谱表示
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

#: 可选谱表示维度（§5.3 模态 B：可选 128 维谱表示）
SPECTRUM_EMBEDDING_DIM: int = 128


def spectrum_statistics(
    spectrum: "pd.DataFrame",
    freq_axis: "pd.Index | None" = None,
) -> "pd.DataFrame":
    """从频谱数据提取可解释统计特征。

    Args:
        spectrum: 频谱矩阵，行 = 时间，列 = 频点。
        freq_axis: 频率刻度；``None`` 时按等间隔处理。

    Returns:
        含峰值频率、带宽、带内/带外能量比等特征的表。

    Raises:
        NotImplementedError: 待 MON-SPAN 数据接入后实现。

    对应开发文档：§6.3 频谱能量变化、§7.3 信号类特征。
    """
    raise NotImplementedError("TODO(§5.3 模态 B): MON-SPAN 接入后实现频谱统计")


def spectrum_embedding(
    spectrum: "pd.DataFrame",
    dim: int = SPECTRUM_EMBEDDING_DIM,
) -> "pd.DataFrame":
    """生成定长谱表示（默认 128 维）。

    Args:
        spectrum: 频谱矩阵。
        dim: 目标维度。

    Returns:
        ``dim`` 列的特征表。

    Raises:
        NotImplementedError: 重采样与归一化策略待确定后实现。

    对应开发文档：§5.3 模态 B、§18.6 模态消融。
    """
    raise NotImplementedError("TODO(§5.3 模态 B): 实现可复现的定长谱重采样")


def spectrum_features(
    spectrum: "pd.DataFrame",
    with_embedding: bool = False,
    dim: int = SPECTRUM_EMBEDDING_DIM,
) -> "pd.DataFrame":
    """聚合频谱特征（统计量 + 可选谱表示）。

    Args:
        spectrum: 频谱矩阵。
        with_embedding: 是否附带定长谱表示。
        dim: 谱表示维度。

    Returns:
        频谱特征表。

    Raises:
        NotImplementedError: 待上游实现。

    对应开发文档：§6.3、§18.6。
    """
    raise NotImplementedError("TODO(§6.3): 聚合频谱统计特征与谱表示")
