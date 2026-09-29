"""M2 观测特征：伪距残差统计与观测自洽性（模态 A 的 ``Res_*`` 列）。

对应开发文档
    §5.3 模态 A（32 PRN × Res）、§6.3 派生特征（伪距/载波/多普勒一致性）、
    §7.3 上下文特征（观测类）、§8.2 S5 观测一致性检测、
    §8.3（观测量不一致 → 首选观测一致性检测）。

职责
    1. 逐历元统计伪距残差（``Res_Gxx``）：有效卫星的均值、最大值、超限个数；
    2. 构造残差相对历史基线的抬升量（欺骗常表现为观测自相矛盾而非整体衰减）；
    3. 输出可按 PRN 追溯的明细，供检测器写入证据（§2.2）。

不做（边界）
    - 不做多普勒 / 载波相位一致性：**当前数据集未提供这两类列**（仅 CNO/Res/Elev），
      相关实现需接入 RXM-RAWX 原始观测后才可进行，见下方 ``NotImplementedError``；
    - 不做阈值判别与结论输出（属 S5 检测器）；
    - 不做卫星几何解算（不在 §1.5 范围内）。

输入 / 输出
    输入：含 ``Res_Gxx`` 与 ``CNO_Gxx`` 列组的特征表
    输出：观测侧特征表（含逐历元统计量与超限计数）

关键约束
    - 残差统计必须先屏蔽无效卫星（复用 ``CNO > 0.5`` 掩码，§5.4 Q3），
      否则“无观测”的 0 值会被误认为“残差极小”；
    - 超限判据的阈值只能来自配置（不硬编码，§2.2）；
    - 基线只使用历史行（``shift(1)``），禁止未来信息（§16.3）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import schema
from src.data.quality import SAT_MASK_CNO_THRESHOLD

#: 默认统计窗口（行数；1 Hz 下等于秒数，§6.2）。
DEFAULT_WINDOW_S: int = 60

#: 输出特征列名。
RES_VALID_MEAN: str = "res_valid_mean"
RES_VALID_MAX: str = "res_valid_max"
RES_VALID_STD: str = "res_valid_std"
RES_OUTLIER_COUNT: str = "res_outlier_count"
RES_BASELINE: str = "res_baseline"
RES_DELTA: str = "res_delta"


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "特征层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def _validate_columns(frame: Any, columns: Sequence[str]) -> None:
    """确认列齐备。

    Raises:
        ValueError: 存在缺失列。
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少列：{missing[:5]}（共 {len(missing)}，见 §5.3 模态 A）")


def pseudorange_residual(
    frame: Any,
    res_columns: Sequence[str] = schema.RES_COLUMNS,
    cno_columns: Sequence[str] = schema.CNO_COLUMNS,
    window_s: int = DEFAULT_WINDOW_S,
    outlier_threshold: float | None = None,
    cno_threshold: float = SAT_MASK_CNO_THRESHOLD,
) -> Any:
    """构造伪距残差统计特征（§6.3、§7.3）。

    Args:
        frame: 特征表。
        res_columns: 残差列组（``Res_Gxx``）。
        cno_columns: C/N0 列组，用于构造有效性掩码。
        window_s: 基线窗口（行数）。
        outlier_threshold: 超限判据阈值；``None`` 表示只统计不判超限（阈值应来自配置）。
        cno_threshold: 卫星有效性阈值（§5.4 Q3）。

    Returns:
        含以下列的 DataFrame：

        - ``res_valid_mean`` / ``res_valid_std`` / ``res_valid_max``：有效卫星残差统计；
        - ``res_outlier_count``：超过阈值且卫星有效的残差个数（阈值为 ``None`` 时全 0）；
        - ``res_baseline`` / ``res_delta``：相对历史基线的抬升量。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列或窗口非法。
    """
    pd = _require_pandas()
    _validate_columns(frame, res_columns)
    _validate_columns(frame, cno_columns)
    if window_s < 1:
        raise ValueError(f"window_s 必须 >= 1：{window_s}")

    res = frame[list(res_columns)]
    # 卫星有效性来自 CNO 列组，而残差来自 Res 列组：两组列名不同，
    # 因此必须**按位置**对齐（转 ndarray）；否则 `&` / `where` 会按列名取并集而错位。
    valid = (frame[list(cno_columns)] > float(cno_threshold)).to_numpy()
    masked = res.where(valid)

    result = pd.DataFrame(index=frame.index)
    result[RES_VALID_MEAN] = masked.mean(axis=1)
    result[RES_VALID_STD] = masked.std(axis=1)
    result[RES_VALID_MAX] = masked.max(axis=1)

    if outlier_threshold is None:
        result[RES_OUTLIER_COUNT] = 0
    else:
        outlier_mask = (res.to_numpy() > float(outlier_threshold)) & valid
        result[RES_OUTLIER_COUNT] = outlier_mask.sum(axis=1).astype("int64")

    baseline = result[RES_VALID_MEAN].rolling(window_s, min_periods=1).mean().shift(1)
    result[RES_BASELINE] = baseline
    result[RES_DELTA] = result[RES_VALID_MEAN] - baseline
    return result


def residual_outliers(
    frame: Any,
    threshold: float,
    res_columns: Sequence[str] = schema.RES_COLUMNS,
    cno_columns: Sequence[str] = schema.CNO_COLUMNS,
    cno_threshold: float = SAT_MASK_CNO_THRESHOLD,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """列出超限残差的明细（PRN 与数值），供检测器写入证据（§2.2）。

    Args:
        frame: 特征表。
        threshold: 超限阈值。
        res_columns: 残差列组。
        cno_columns: C/N0 列组。
        cno_threshold: 卫星有效性阈值。
        limit: 最多返回多少条（``None`` 表示不限）。

    Returns:
        形如 ``{"prn": "G03", "residual": 12.3}`` 的记录列表，按残差降序。
        若提供了时间列，记录中会附带 ``timestamp``。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 缺少列。
    """
    _require_pandas()
    _validate_columns(frame, res_columns)
    _validate_columns(frame, cno_columns)

    timestamp = frame[schema.TIME_COLUMN].iloc[0] if schema.TIME_COLUMN in frame.columns else None
    records: list[dict[str, Any]] = []
    for res_column, cno_column in zip(res_columns, cno_columns):
        value = frame[res_column].iloc[-1]
        cno = frame[cno_column].iloc[-1]
        if cno is None or float(cno) <= float(cno_threshold):
            continue  # 该卫星无有效观测
        if value is not None and float(value) > float(threshold):
            prn = res_column.split("_")[-1]
            record: dict[str, Any] = {"prn": prn, "residual": float(value)}
            if timestamp is not None:
                record["timestamp"] = str(timestamp)
            records.append(record)

    records.sort(key=lambda item: item["residual"], reverse=True)
    return records if limit is None else records[:limit]


def observation_features(
    frame: Any,
    window_s: int = DEFAULT_WINDOW_S,
    outlier_threshold: float | None = None,
) -> Any:
    """聚合观测侧特征。

    Args:
        frame: 特征表。
        window_s: 基线窗口（行数）。
        outlier_threshold: 超限阈值（来自配置）。

    Returns:
        观测侧特征表。

    Raises:
        ImportError: 未安装 pandas。
    """
    pd = _require_pandas()
    return pd.concat(
        [pseudorange_residual(frame, window_s=window_s, outlier_threshold=outlier_threshold)],
        axis=1,
    )


def doppler_consistency(*args: Any, **kwargs: Any) -> Any:
    """多普勒一致性特征实现入口。

    Raises:
        NotImplementedError: 当前数据集未提供多普勒观测列（§5.3 模态 C 的 RAWX 未接入），
            接入 ``RXM-RAWX`` 后应在此实现多普勒与伪距变化率的一致性判据（§6.3）。
    """
    raise NotImplementedError(
        "TODO(§5.3 模态 C): 当前数据集无多普勒列；需接入 RXM-RAWX 原始观测"
    )


def carrier_consistency(*args: Any, **kwargs: Any) -> Any:
    """载波相位一致性特征实现入口。

    Raises:
        NotImplementedError: 当前数据集未提供载波相位列，且周跳判据需另行标定（§6.3）。
    """
    raise NotImplementedError(
        "TODO(§5.3 模态 C): 当前数据集无载波相位列；需接入 RXM-RAWX 原始观测"
    )


def feature_columns() -> Mapping[str, tuple[str, ...]]:
    """返回本模块产出的特征列名分组。"""
    return {
        "residual": (
            RES_VALID_MEAN,
            RES_VALID_STD,
            RES_VALID_MAX,
            RES_OUTLIER_COUNT,
            RES_BASELINE,
            RES_DELTA,
        )
    }
