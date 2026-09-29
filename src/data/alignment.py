"""M1 时间排序、多流对齐与缺口报告。

对应开发文档
    §5.4 Q1 时间顺序、§5.4 Q2 缺失时间点、§6.1 标准处理流水线（第 3–4 步）、
    §16.3 防泄漏规范（对齐不得引入未来信息）。

职责
    1. 按时间戳稳定排序（``recordTime`` 可能非严格有序，§5.4 Q1）；
    2. 检测并处理重复时间戳；
    3. 把多源流对齐到统一时间网格（当前数据集已逐历元对齐，该能力供后续多源接入）；
    4. 输出缺口报告（缺口数、最大缺口、缺失占比），供数据质量评估与 §17 指标使用。

不做（边界）
    - 不做缺失值填补（填充 / 插值 / mask-aware 由模型侧选择，§5.4 Q2）；
    - 不做特征构造与标准化（→ ``src/features/*``、§6.4）；
    - 不使用未来历元参与对齐（§16.3：只允许向前取值）。

输入 / 输出
    输入：DataFrame 与时间列名
    输出：排序 / 对齐后的 DataFrame、AlignmentReport

关键约束
    - 排序必须稳定，保证同输入同输出（§20.4 可复现）；
    - 对齐方向固定为 ``backward``（只用当前及过去历元），避免时间泄漏；
    - 时间列必须是已解析的时间类型，否则直接报错（不静默降级）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.data import schema

#: 默认对齐容差（秒）。一期数据为 1 Hz，容差小于 1 s 以保证不跨越相邻历元。
DEFAULT_TOLERANCE_S: float = 0.5

#: 判定“时间断裂”的步长倍数阈值：相邻间隔超过 ``freq_s × 该倍数`` 记为一次缺口。
GAP_STEP_FACTOR: float = 1.5


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "数据层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def require_datetime(frame: Any, column: str) -> Any:
    """确认时间列为时间类型并返回该列。

    Raises:
        ValueError: 列缺失或不是时间类型（应先调用 ``parser.standardize_timestamp``）。
    """
    pd = _require_pandas()
    if column not in frame.columns:
        raise ValueError(f"缺少时间列 {column!r}")
    column_data = frame[column]
    if not pd.api.types.is_datetime64_any_dtype(column_data):
        raise ValueError(
            f"时间列 {column!r} 尚未解析为时间类型，请先调用 "
            "src.data.parser.standardize_timestamp()（§6.1 第 2 步）"
        )
    return column_data


@dataclass(slots=True)
class AlignmentReport:
    """对齐与缺口质量报告。

    Attributes:
        rows: 记录行数。
        streams: 参与对齐的数据流数量。
        start: 起始时间（字符串，便于写入实验记录）。
        end: 结束时间。
        span_s: 时间跨度（秒）。
        expected_rows: 按采样率推算的应有行数。
        missing_rows: 缺失行数（推算值）。
        missing_ratio: 缺失占比。
        max_gap_s: 最大相邻间隔（秒），对应 §5.4 Q2 的缺口问题。
        gap_count: 断裂次数（相邻间隔超过 ``freq_s × GAP_STEP_FACTOR``）。
        duplicate_rows: 重复时间戳行数。
        details: 分项统计。
    """

    rows: int = 0
    streams: int = 1
    start: str | None = None
    end: str | None = None
    span_s: float = 0.0
    expected_rows: int = 0
    missing_rows: int = 0
    missing_ratio: float = 0.0
    max_gap_s: float = 0.0
    gap_count: int = 0
    duplicate_rows: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        """转为可序列化映射（供实验记录与数据质量评估使用）。"""
        return {
            "rows": self.rows,
            "streams": self.streams,
            "start": self.start,
            "end": self.end,
            "span_s": self.span_s,
            "expected_rows": self.expected_rows,
            "missing_rows": self.missing_rows,
            "missing_ratio": self.missing_ratio,
            "max_gap_s": self.max_gap_s,
            "gap_count": self.gap_count,
            "duplicate_rows": self.duplicate_rows,
            "details": dict(self.details),
        }


def sort_by_time(frame: Any, column: str = schema.TIME_COLUMN) -> Any:
    """按时间戳稳定排序（§5.4 Q1）。

    Args:
        frame: 输入表（时间列须已解析为时间类型）。
        column: 时间列名。

    Returns:
        排序并重置索引后的新表。

    Raises:
        ValueError: 时间列缺失或类型不对。
    """
    require_datetime(frame, column)
    return frame.sort_values(column, kind="stable").reset_index(drop=True)


def drop_duplicate_timestamps(
    frame: Any,
    column: str = schema.TIME_COLUMN,
    keep: str = "last",
) -> Any:
    """去除重复时间戳（同一历元只保留一条）。

    Args:
        frame: 输入表。
        column: 时间列名。
        keep: 保留策略，传给 ``DataFrame.drop_duplicates``。

    Returns:
        去重后的新表（索引重置）。

    Raises:
        ValueError: 时间列缺失或类型不对。
    """
    require_datetime(frame, column)
    return frame.drop_duplicates(subset=[column], keep=keep).reset_index(drop=True)


def align_streams(
    streams: Mapping[str, Any],
    target_column: str = schema.TIME_COLUMN,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    direction: str = "backward",
) -> tuple[Any, AlignmentReport]:
    """把多源数据流对齐到统一时间网格。

    Args:
        streams: 流名称 → 数据表（每个表都须含已解析的时间列）。
        target_column: 对齐依据的时间列。
        tolerance_s: 允许的时间容差（秒）。
        direction: ``merge_asof`` 的取值方向；默认 ``"backward"``，
            即只使用当前及过去历元的值，避免引入未来信息（§16.3）。

    Returns:
        ``(对齐后的 DataFrame, AlignmentReport)``。

    Raises:
        ValueError: 输入为空，或时间列未解析。
        KeyError: 未知的 ``direction``。

    Note:
        当前数据集已是逐历元一行（单流），本函数供后续接入 RAWX / MON-SPAN
        等多采样率数据时使用。
    """
    pd = _require_pandas()
    if not streams:
        raise ValueError("streams 不能为空")
    if direction not in ("backward", "forward", "nearest"):
        raise KeyError(f"direction 需为 backward/forward/nearest，实际为 {direction!r}")

    names = list(streams)
    base = sort_by_time(streams[names[0]], target_column)
    for name in names[1:]:
        other = sort_by_time(streams[name], target_column)
        base = pd.merge_asof(
            base,
            other,
            on=target_column,
            direction=direction,
            tolerance=pd.Timedelta(seconds=tolerance_s),
            suffixes=("", f"__{name}"),
        )

    report = alignment_report(base, target_column)
    report.streams = len(names)
    return base, report


def alignment_report(
    frame: Any,
    column: str = schema.TIME_COLUMN,
    freq_s: float = 1.0,
) -> AlignmentReport:
    """统计时间缺口与重复情况（§5.4 Q2）。

    Args:
        frame: 输入表。
        column: 时间列名。
        freq_s: 期望采样间隔（秒），一期正常数据为 1 Hz。

    Returns:
        AlignmentReport。

    Raises:
        ValueError: 时间列缺失、类型不对，或 ``freq_s`` 非正。
    """
    pd = _require_pandas()
    if freq_s <= 0:
        raise ValueError(f"freq_s 必须为正：{freq_s}")

    times = require_datetime(frame, column)
    if times.empty:
        return AlignmentReport()

    ordered = times.sort_values()
    deltas = ordered.diff().dt.total_seconds().dropna()

    span_s = float((ordered.iloc[-1] - ordered.iloc[0]).total_seconds())
    expected_rows = int(round(span_s / freq_s)) + 1
    missing_rows = max(0, expected_rows - int(len(ordered)))
    gaps = deltas[deltas > freq_s * GAP_STEP_FACTOR]

    return AlignmentReport(
        rows=int(len(ordered)),
        start=str(ordered.iloc[0]),
        end=str(ordered.iloc[-1]),
        span_s=span_s,
        expected_rows=expected_rows,
        missing_rows=missing_rows,
        missing_ratio=(missing_rows / expected_rows) if expected_rows else 0.0,
        max_gap_s=float(deltas.max()) if not deltas.empty else 0.0,
        gap_count=int(len(gaps)),
        duplicate_rows=int(times.duplicated().sum()),
        details={
            "freq_s": freq_s,
            "gap_step_factor": GAP_STEP_FACTOR,
            "median_step_s": float(deltas.median()) if not deltas.empty else 0.0,
            "max_gap_seconds_total": float(gaps.sum()) if not gaps.empty else 0.0,
        },
    )


def missing_timestamps(
    frame: Any,
    column: str = schema.TIME_COLUMN,
    freq_s: float = 1.0,
) -> Sequence[Any]:
    """列出缺失的时间点（§5.4 Q2）。

    Args:
        frame: 输入表。
        column: 时间列名。
        freq_s: 期望采样间隔（秒）。

    Returns:
        缺失的时间点序列（``DatetimeIndex``）。

    Raises:
        ValueError: 时间列缺失或类型不对。
    """
    pd = _require_pandas()
    times = require_datetime(frame, column)
    if times.empty:
        return pd.DatetimeIndex([])

    ordered = times.sort_values()
    full_grid = pd.date_range(ordered.iloc[0], ordered.iloc[-1], freq=f"{freq_s}s")
    return full_grid.difference(pd.DatetimeIndex(ordered))
