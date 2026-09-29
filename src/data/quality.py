"""M1 数据质量：卫星有效性掩码、缺失掩码与数据质量评分。

对应开发文档
    §5.4 Q2 缺失时间点、§5.4 Q3 卫星特征稀疏（``sat_mask = (CNO > 0.5)``）、
    §7.2 Context 组成的 Q_t（Quality Context）、§7.3 质量类上下文特征、
    §12.2 融合权重（数据质量参与加权）、§17.5 系统级指标。

职责
    1. 逐历元标识**有效卫星数**（``CNO > 0.5``）并据此生成 ``sat_mask``；
    2. 生成 ``miss_mask``（时间断裂标记）与缺口时长；
    3. 计算可解释的数据质量评分（0–1），供 Context 与融合权重使用。

不做（边界）
    - 不做填充 / 插值（由模型侧决定，§5.4 Q2）；
    - 不做异常值剔除（属检测器职责，§8.2 S1）；
    - 不改写原始观测值，只新增掩码与质量列。

输入 / 输出
    输入：特征表（含 CNO 列组与时间列）
    输出：附加了掩码列的 DataFrame 与 [0, 1] 区间的质量评分

关键约束
    - 卫星掩码阈值集中在本模块定义（§2.3 可验证）；
    - 质量评分只使用可解释分项，禁止黑盒打分（§2.2）；
    - 无卫星历元不得被当作“低 C/N0”（§5.4 Q3），必须在掩码中显式区分。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import alignment, schema

#: C/N0 有效性判据（§5.4 Q3：sat_mask = (CNO > 0.5)）。
SAT_MASK_CNO_THRESHOLD: float = 0.5

#: 掩码列名。
SAT_MASK_COLUMN: str = "sat_mask"
VALID_SAT_COUNT_COLUMN: str = "valid_sat_count"
MISS_MASK_COLUMN: str = "miss_mask"
GAP_BEFORE_COLUMN: str = "gap_before_s"
QUALITY_COLUMN: str = "data_quality"

#: 质量评分分项权重（归一到 1），只用可解释分项，便于在 §17.5 复核。
QUALITY_WEIGHTS: Mapping[str, float] = {
    "completeness": 0.4,        # 时间完整度（1 − 缺失率）
    "satellite_coverage": 0.3,  # 有效卫星数相对基线
    "field_validity": 0.3,      # PVT 关键字段非缺失比例
}

#: 参与“卫星覆盖度”评估的基线卫星数（正常数据实测约 8–12 颗，取 8 作为参考下限）。
BASELINE_SAT_COUNT: int = 8


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "数据层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def sat_mask(cno: Any, threshold: float = SAT_MASK_CNO_THRESHOLD) -> Any:
    """按 §5.4 Q3 生成单列卫星有效性掩码。

    该判据由开发文档明确给出（``sat_mask = (CNO > 0.5)``），因此直接实现。

    Args:
        cno: C/N0 序列（无卫星处允许为 0 或 NaN）。
        threshold: 判定阈值。

    Returns:
        与输入等长的布尔序列。
    """
    return cno > threshold


def valid_satellite_count(
    frame: Any,
    cno_columns: Sequence[str] = schema.CNO_COLUMNS,
    threshold: float = SAT_MASK_CNO_THRESHOLD,
) -> Any:
    """统计每个历元的有效卫星数（§5.4 Q3）。

    Args:
        frame: 特征表。
        cno_columns: C/N0 列组。
        threshold: 卫星掩码阈值。

    Returns:
        每行有效卫星数的整型序列。
    """
    missing = [name for name in cno_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少 C/N0 列：{missing[:5]}（共 {len(missing)}）")
    return (frame[list(cno_columns)] > threshold).sum(axis=1).astype("int64")


def attach_masks(
    frame: Any,
    threshold: float = SAT_MASK_CNO_THRESHOLD,
    freq_s: float = 1.0,
    time_column: str = schema.TIME_COLUMN,
) -> Any:
    """给特征表附加掩码列（§5.4 Q2、Q3、§6.1 流水线的 mask 步骤）。

    新增列：

    - ``valid_sat_count``：该历元有效卫星数；
    - ``sat_mask``：有效卫星数 > 0（即是否存在可用观测）；
    - ``gap_before_s``：与上一历元的时间差（秒），首行为 0；
    - ``miss_mask``：该历元之前是否存在时间断裂（``gap_before_s > freq_s``）。

    Args:
        frame: 特征表（时间列须已解析）。
        threshold: 卫星掩码阈值。
        freq_s: 期望采样间隔（秒）。
        time_column: 时间列名。

    Returns:
        含上述四列的新表。

    Raises:
        ValueError: 缺少 C/N0 列或时间列未解析。
    """
    _require_pandas()
    result = frame.copy()
    result[VALID_SAT_COUNT_COLUMN] = valid_satellite_count(result, threshold=threshold)
    result[SAT_MASK_COLUMN] = result[VALID_SAT_COUNT_COLUMN] > 0

    times = alignment.require_datetime(result, time_column)
    # 先按时间排序求相邻差，再按原索引回填，保证输出行序与输入一致
    gaps = times.sort_values().diff().dt.total_seconds().reindex(result.index)
    result[GAP_BEFORE_COLUMN] = gaps.fillna(0.0).astype("float64")
    result[MISS_MASK_COLUMN] = result[GAP_BEFORE_COLUMN] > float(freq_s)
    return result


def miss_mask(
    frame: Any,
    freq_s: float = 1.0,
    time_column: str = schema.TIME_COLUMN,
) -> Any:
    """生成时间断裂掩码（§5.4 Q2）。

    Args:
        frame: 特征表（时间列须已解析）。
        freq_s: 期望采样间隔（秒）。
        time_column: 时间列名。

    Returns:
        布尔序列：True 表示该历元之前存在缺口。

    Raises:
        ValueError: 时间列未解析。
    """
    times = alignment.require_datetime(frame, time_column)
    ordered = times.sort_values()
    flags = ordered.diff().dt.total_seconds().fillna(0.0) > float(freq_s)
    return flags.reindex(frame.index).fillna(False).astype(bool)


def _validity_ratio(frame: Any, columns: Sequence[str]) -> float:
    """计算给定列组在整表范围内的非空且非 NaN 比例。"""
    present = [name for name in columns if name in frame.columns]
    if not present:
        return 0.0
    block = frame[present]
    if block.empty:
        return 0.0
    return float(block.notna().to_numpy().mean())


def data_quality_score(
    frame: Any,
    weights: Mapping[str, float] = QUALITY_WEIGHTS,
    freq_s: float = 1.0,
    time_column: str = schema.TIME_COLUMN,
    baseline_sat_count: int = BASELINE_SAT_COUNT,
) -> float:
    """计算数据质量评分（0–1），供 Context 与融合权重使用。

    分项（均可从数据复算，§2.2）：

    - ``completeness``：1 − 时间缺失占比（由 ``alignment_report`` 给出）；
    - ``satellite_coverage``：平均有效卫星数 / 基线卫星数（截断到 1）；
    - ``field_validity``：PVT 关键字段的非空比例。

    Args:
        frame: 特征表（时间列须已解析）。
        weights: 分项权重。
        freq_s: 期望采样间隔（秒）。
        time_column: 时间列名。
        baseline_sat_count: 卫星覆盖度基线（颗）。

    Returns:
        质量评分，取值 [0, 1]。

    Raises:
        ValueError: 权重之和为 0，或基线卫星数非正。
    """
    if baseline_sat_count <= 0:
        raise ValueError(f"baseline_sat_count 必须为正：{baseline_sat_count}")

    report = alignment.alignment_report(frame, column=time_column, freq_s=freq_s)
    completeness = 1.0 - float(report.missing_ratio)

    counts = valid_satellite_count(frame)
    coverage = min(1.0, float(counts.mean()) / baseline_sat_count) if len(counts) else 0.0

    validity = _validity_ratio(frame, schema.PVT_COLUMNS)

    components = {
        "completeness": max(0.0, min(1.0, completeness)),
        "satellite_coverage": max(0.0, min(1.0, coverage)),
        "field_validity": max(0.0, min(1.0, validity)),
    }
    total_weight = sum(float(weights.get(name, 0.0)) for name in components)
    if total_weight <= 0:
        raise ValueError(f"quality 权重之和必须为正：{dict(weights)}")

    score = sum(components[name] * float(weights.get(name, 0.0)) for name in components) / total_weight
    return max(0.0, min(1.0, score))


def quality_breakdown(
    frame: Any,
    freq_s: float = 1.0,
    baseline_sat_count: int = BASELINE_SAT_COUNT,
) -> Mapping[str, Any]:
    """返回质量评分的分项明细（便于写实验记录与排查）。

    Args:
        frame: 特征表。
        freq_s: 期望采样间隔（秒）。
        baseline_sat_count: 卫星覆盖度基线。

    Returns:
        含 ``score`` 与各分项的映射。
    """
    report = alignment.alignment_report(frame, freq_s=freq_s)
    counts = valid_satellite_count(frame)
    components = {
        "completeness": 1.0 - float(report.missing_ratio),
        "satellite_coverage": min(1.0, float(counts.mean()) / baseline_sat_count) if len(counts) else 0.0,
        "field_validity": _validity_ratio(frame, schema.PVT_COLUMNS),
    }
    return {
        "score": data_quality_score(frame, freq_s=freq_s, baseline_sat_count=baseline_sat_count),
        "components": components,
        "mean_valid_sat_count": float(counts.mean()) if len(counts) else 0.0,
        "missing_ratio": float(report.missing_ratio),
        "max_gap_s": float(report.max_gap_s),
    }
