"""M2 导航特征：PVT 精度、DOP 与钟差差分（模态 A）。

对应开发文档
    §5.3 模态 A（PVT 特征与全局状态特征）、§6.3 派生特征（PVT 变化率、
    ``clkB`` / ``clkD`` 一阶差分、PVT/DOP 滑动统计）、§7.3 上下文特征（导航类）、
    §8.2 S6 PVT/DOP 检测。

职责
    1. 构造精度类指标（``hAcc`` / ``vAcc`` / ``tAcc``）与地速的变化率；
    2. 构造 DOP（``pDOP`` / ``tDOP`` / ``hDOP``）的滑动统计与相对基线变化；
    3. 构造 ``clkB`` / ``clkD`` 一阶差分（§6.3 明确要求）。

不做（边界）
    - 不做卫星层异常判定（属 S4，§8.2）；
    - 不做观测一致性分析（→ ``src/features/observation.py``）；
    - 不做阈值判别与结论输出（属检测器）。

输入 / 输出
    输入：含 PVT 列组的特征表
    输出：导航侧特征表

关键约束
    - 变化率与基线**只使用当前及历史行**（``shift(1)``），禁止未来信息（§2.5、§16.3）；
    - 单位与列名保持与原始列一致（如 ``hAcc`` 的差分仍为“精度单位/行”），
      不做不可追溯的归一化（§2.2）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import schema

#: 滑动统计窗口（行数；1 Hz 下等于秒数，§6.2 默认 60 s）。
DEFAULT_STAT_WINDOW_S: int = 60

#: 参与差分与统计的列组。
ACCURACY_COLUMNS: tuple[str, ...] = ("hAcc", "vAcc", "tAcc")
DOP_COLUMNS: tuple[str, ...] = ("pDOP", "tDOP", "hDOP")
CLOCK_COLUMNS: tuple[str, ...] = ("clkB", "clkD")
SPEED_COLUMNS: tuple[str, ...] = ("gSpeed",)


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "特征层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def _require_columns(frame: Any, columns: Sequence[str]) -> None:
    """确认列齐备。

    Raises:
        ValueError: 存在缺失列。
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}（见 §5.3 模态 A）")


def _delta_columns(
    frame: Any,
    columns: Sequence[str],
    suffix: str = "_delta",
) -> dict[str, Any]:
    """对给定列组求一阶差分（当前历元 − 上一历元）。"""
    return {f"{name}{suffix}": frame[name].astype("float64").diff() for name in columns}


def _baseline_delta_columns(
    frame: Any,
    columns: Sequence[str],
    window_s: int,
    suffix: str = "_vs_baseline",
) -> dict[str, Any]:
    """对给定列组求“当前值 − 历史窗口均值”。"""
    result: dict[str, Any] = {}
    for name in columns:
        series = frame[name].astype("float64")
        baseline = series.rolling(window_s, min_periods=1).mean().shift(1)
        result[f"{name}{suffix}"] = series - baseline
    return result


def pvt_features(
    frame: Any,
    window_s: int = DEFAULT_STAT_WINDOW_S,
    columns: Sequence[str] = (*ACCURACY_COLUMNS, *SPEED_COLUMNS),
) -> Any:
    """构造 PVT 精度与速度的变化特征（§6.3）。

    Args:
        frame: 特征表。
        window_s: 基线窗口（行数）。
        columns: 参与计算的列组。

    Returns:
        含 ``*_delta`` 与 ``*_vs_baseline`` 列的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列。
    """
    pd = _require_pandas()
    _require_columns(frame, columns)
    result = pd.DataFrame(index=frame.index)
    for key, value in _delta_columns(frame, columns).items():
        result[key] = value
    for key, value in _baseline_delta_columns(frame, columns, window_s).items():
        result[key] = value
    return result


def dop_features(
    frame: Any,
    window_s: int = DEFAULT_STAT_WINDOW_S,
    columns: Sequence[str] = DOP_COLUMNS,
) -> Any:
    """构造 DOP 的滑动统计与相对基线变化（§6.3）。

    Args:
        frame: 特征表。
        window_s: 滑动窗口（行数）。
        columns: DOP 列组。

    Returns:
        含 ``*_mean`` / ``*_std`` / ``*_vs_baseline`` 列的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列。
    """
    pd = _require_pandas()
    _require_columns(frame, columns)
    result = pd.DataFrame(index=frame.index)
    for name in columns:
        series = frame[name].astype("float64")
        rolling = series.rolling(window_s, min_periods=1)
        result[f"{name}_mean"] = rolling.mean()
        result[f"{name}_std"] = rolling.std()
        result[f"{name}_vs_baseline"] = series - rolling.mean().shift(1)
    return result


def clock_features(
    frame: Any,
    columns: Sequence[str] = CLOCK_COLUMNS,
) -> Any:
    """构造 ``clkB`` / ``clkD`` 一阶差分（§6.3 明确要求）。

    Args:
        frame: 特征表。
        columns: 钟差列组。

    Returns:
        含 ``clkB_diff`` / ``clkD_diff`` 的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列。
    """
    pd = _require_pandas()
    _require_columns(frame, columns)
    result = pd.DataFrame(index=frame.index)
    for name in columns:
        result[f"{name}_diff"] = frame[name].astype("float64").diff()
    return result


def navigation_features(
    frame: Any,
    window_s: int = DEFAULT_STAT_WINDOW_S,
) -> Any:
    """聚合导航侧特征（PVT + DOP + 钟差）。

    Args:
        frame: 特征表。
        window_s: 滑动窗口（行数）。

    Returns:
        导航侧特征表。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列。
    """
    pd = _require_pandas()
    return pd.concat(
        [
            pvt_features(frame, window_s=window_s),
            dop_features(frame, window_s=window_s),
            clock_features(frame),
        ],
        axis=1,
    )


def feature_columns() -> Mapping[str, tuple[str, ...]]:
    """返回本模块产出的特征列名分组。"""
    return {
        "pvt": tuple(
            f"{name}{suffix}"
            for name in (*ACCURACY_COLUMNS, *SPEED_COLUMNS)
            for suffix in ("_delta", "_vs_baseline")
        ),
        "dop": tuple(
            f"{name}{suffix}" for name in DOP_COLUMNS for suffix in ("_mean", "_std", "_vs_baseline")
        ),
        "clock": tuple(f"{name}_diff" for name in CLOCK_COLUMNS),
    }
