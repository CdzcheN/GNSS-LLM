"""M2 信号质量特征：C/N0 统计与变化率（模态 A）。

对应开发文档
    §5.3 模态 A（32 PRN × CNO）、§6.3 派生特征（C/N0 变化率、可见卫星数量变化率）、
    §7.3 上下文特征（信号类）、§8.2 S3 C/N0 检测、§8.3（C/N0 整体下降 → 压制）。

职责
    1. 按 §5.4 Q3 的 ``sat_mask`` 屏蔽无效卫星后统计 C/N0（均值、离散度、有效星数）；
    2. 构造 C/N0 相对历史基线的变化量（压制干扰的主要征兆）；
    3. 构造可见卫星数量变化率（§6.3 明确要求）。

不做（边界）
    - 不做卫星有效性判定（阈值来自 ``src/data/quality.py``）；
    - 不做频谱与 AGC 特征（模态 B 数据未提供，见下方说明）；
    - 不做阈值判别与结论输出（属检测器，§8.2 S3）。

输入 / 输出
    输入：含 ``CNO_Gxx`` 列组的特征表（时间列须已排序）
    输出：信号侧特征表（列名固定，便于 Context 与检测器复用）

关键约束
    - 变化率**只使用当前及历史行**（``shift(1)``），禁止未来信息（§2.5、§16.3）；
    - 无效卫星（``CNO <= 0.5``）必须先屏蔽再统计，否则“无观测”会被当成“弱信号”（§5.4 Q3）；
    - 滚动窗口按行数近似秒数（数据为 1 Hz），存在缺口时误差已在质量列中标识。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import schema
from src.data.quality import SAT_MASK_CNO_THRESHOLD

#: 默认变化率窗口（秒，§6.2 的 W = 60 s）。
DEFAULT_DELTA_WINDOW_S: int = 60

#: 输出特征列名。
CN0_VALID_MEAN: str = "cn0_valid_mean"
CN0_VALID_STD: str = "cn0_valid_std"
CN0_VALID_COUNT: str = "cn0_valid_count"
CN0_BASELINE: str = "cn0_baseline"
CN0_DELTA: str = "cn0_delta_db"
SAT_COUNT_DELTA: str = "sat_count_delta"
AVG_CNO_DELTA: str = "avg_cno_delta"


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "特征层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def cn0_features(
    frame: Any,
    cno_columns: Sequence[str] = schema.CNO_COLUMNS,
    window_s: int = DEFAULT_DELTA_WINDOW_S,
    threshold: float = SAT_MASK_CNO_THRESHOLD,
) -> Any:
    """构造 C/N0 统计与变化率特征（§6.3、§7.3）。

    Args:
        frame: 特征表。
        cno_columns: C/N0 列组。
        window_s: 变化率基线窗口（行数，1 Hz 下等于秒数）。
        threshold: 卫星有效性阈值（§5.4 Q3）。

    Returns:
        含以下列的 DataFrame（索引与输入一致）：

        - ``cn0_valid_mean`` / ``cn0_valid_std`` / ``cn0_valid_count``：屏蔽无效卫星后的统计；
        - ``cn0_baseline``：历史窗口均值（不含当前行）；
        - ``cn0_delta_db``：当前均值相对基线的变化（dB，负值表示下降）。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少 C/N0 列或没有有效卫星列。
    """
    pd = _require_pandas()
    missing = [name for name in cno_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少 C/N0 列：{missing[:5]}（共 {len(missing)}）")
    if window_s < 1:
        raise ValueError(f"window_s 必须 >= 1：{window_s}")

    cno = frame[list(cno_columns)]
    valid = cno > float(threshold)
    masked = cno.where(valid)

    result = pd.DataFrame(index=frame.index)
    result[CN0_VALID_COUNT] = valid.sum(axis=1).astype("int64")
    result[CN0_VALID_MEAN] = masked.mean(axis=1)
    result[CN0_VALID_STD] = masked.std(axis=1)

    # 基线只使用历史行（shift(1)），确保不引入未来信息
    baseline = result[CN0_VALID_MEAN].rolling(window_s, min_periods=1).mean().shift(1)
    result[CN0_BASELINE] = baseline
    result[CN0_DELTA] = result[CN0_VALID_MEAN] - baseline
    return result


def satellite_count_features(
    frame: Any,
    count_column: str = "NumSats",
    window_s: int = DEFAULT_DELTA_WINDOW_S,
) -> Any:
    """构造可见卫星数量变化特征（§6.3）。

    Args:
        frame: 特征表（含 ``NumSats``）。
        count_column: 卫星数量列名。
        window_s: 基线窗口（行数）。

    Returns:
        含 ``sat_count_delta`` 的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少卫星数量列。
    """
    pd = _require_pandas()
    if count_column not in frame.columns:
        raise ValueError(f"缺少卫星数量列 {count_column!r}（§5.3 模态 A）")

    counts = frame[count_column].astype("float64")
    baseline = counts.rolling(window_s, min_periods=1).mean().shift(1)
    result = pd.DataFrame(index=frame.index)
    result[SAT_COUNT_DELTA] = counts - baseline

    if "AvgCNO" in frame.columns:
        avg = frame["AvgCNO"].astype("float64")
        result[AVG_CNO_DELTA] = avg - avg.rolling(window_s, min_periods=1).mean().shift(1)
    return result


def signal_features(
    frame: Any,
    window_s: int = DEFAULT_DELTA_WINDOW_S,
    threshold: float = SAT_MASK_CNO_THRESHOLD,
) -> Any:
    """聚合信号侧特征（C/N0 + 卫星数量 + AvgCNO）。

    Args:
        frame: 特征表。
        window_s: 变化率窗口（行数）。
        threshold: 卫星有效性阈值。

    Returns:
        信号侧特征表（列见 ``cn0_features`` 与 ``satellite_count_features``）。

    Raises:
        ImportError: 未安装 pandas。
    """
    pd = _require_pandas()
    return pd.concat(
        [
            cn0_features(frame, window_s=window_s, threshold=threshold),
            satellite_count_features(frame, window_s=window_s),
        ],
        axis=1,
    )


def agc_features(*args: Any, **kwargs: Any) -> Any:
    """AGC 特征（模态 B）实现入口。

    Raises:
        NotImplementedError: 当前数据集**未提供** AGC / 频谱列（仅有模态 A 的
            C/N0、Res、Elev 与 PVT 列），因此无法实现。接入 MON-SPAN 数据后，
            应在此实现 AGC 基线与偏移特征（§5.3 模态 B、§7.3 信号类特征）。
    """
    raise NotImplementedError(
        "TODO(§5.3 模态 B): 当前数据集无 AGC/频谱列；需接入 MON-SPAN 数据"
    )


def feature_columns() -> Mapping[str, tuple[str, ...]]:
    """返回本模块产出的特征列名，便于上游按需选取。"""
    return {
        "cn0": (CN0_VALID_MEAN, CN0_VALID_STD, CN0_VALID_COUNT, CN0_BASELINE, CN0_DELTA),
        "satellite": (SAT_COUNT_DELTA, AVG_CNO_DELTA),
    }
