"""M1 数据接入与解析：读取数据集、校验 schema、标准化时间戳。

对应开发文档
    §5.1 原始数据、§5.2 已提取数据、§5.3 模态定义、§6.1 标准处理流水线（第 1–2 步）。

职责
    1. 读取 ``data/`` 下的特征表（当前数据集为 CSV 形式），并用
       ``src/data/schema.py`` 校验列名与顺序；
    2. 把 ``Timestamp`` 列解析为标准时间类型（不做排序，排序属 alignment）；
    3. 按 §16.2 的日期划分切分数据（训练 / 验证 / 留出）；
    4. 支持分块迭代，避免一次性把 1.6M 行全部载入内存。

不做（边界）
    - 不做排序与多流对齐（→ ``src/data/alignment.py``）；
    - 不做卫星掩码与质量评分（→ ``src/data/quality.py``）；
    - 不做特征构造（→ ``src/features/*``）与标准化（§6.4 统计量只能来自训练集）。

输入 / 输出
    输入：CSV 路径、可选行数上限
    输出：pandas DataFrame（列与 §5.3 一致，另附 ``source_file`` 溯源列）

关键约束
    - 时间戳解析失败必须报错，不得静默产生 NaT（否则缺口统计会被污染）；
    - 列名校验必须严格（顺序也是约定的一部分，见 ``schema.assert_columns``）；
    - 本模块顶层不导入 pandas：未安装时模块仍可导入，只在调用时提示安装。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.data import schema

#: 数据文件相对于仓库根的位置校验：调用方传入的路径不存在时给出可读错误。
#: 逐块读取的默认行数（约 20 万行，兼顾内存与吞吐）。
DEFAULT_CHUNKSIZE: int = 200_000

#: 溯源列名（附加列，不属于 §5.3 的 112 列 schema）。
SOURCE_COLUMN: str = "source_file"


def _require_pandas() -> Any:
    """惰性导入 pandas。

    Returns:
        ``pandas`` 模块。

    Raises:
        ImportError: 未安装 pandas（依赖见 requirements.txt）。
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "数据层需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def resolve_path(name: str, dataset_dir: str | Path = schema.DATASET_DIR) -> Path:
    """把数据集文件名解析为完整路径。

    Args:
        name: 文件名（如 ``gnss_complete_featuresObsSatPvt_12-16.csv``）或直接路径。
        dataset_dir: 数据集目录。

    Returns:
        存在的文件路径。

    Raises:
        FileNotFoundError: 文件不存在（消息中给出期望路径）。
    """
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = Path(dataset_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}（数据集目录：{dataset_dir}）")
    return path


def standardize_timestamp(
    frame: Any,
    column: str = schema.TIME_COLUMN,
    fmt: str = schema.TIMESTAMP_FORMAT,
) -> Any:
    """把时间戳列解析为标准时间类型（不排序）。

    Args:
        frame: 输入表。
        column: 时间戳列名。
        fmt: 时间格式，默认与实测表头一致（``%Y-%m-%d %H:%M:%S``）。

    Returns:
        新的 DataFrame（时间列已转换）。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 存在无法解析的时间戳。
    """
    pd = _require_pandas()
    if column not in frame.columns:
        raise ValueError(f"缺少时间列 {column!r}")

    converted = frame.copy()
    try:
        converted[column] = pd.to_datetime(converted[column], format=fmt, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"时间戳解析失败（期望格式 {fmt}）：{exc}") from exc
    return converted


def load_features_csv(
    name: str,
    dataset_dir: str | Path = schema.DATASET_DIR,
    nrows: int | None = None,
    usecols: Sequence[str] | None = None,
    validate: bool = True,
    parse_time: bool = True,
) -> Any:
    """读取单个特征表并对齐 schema。

    Args:
        name: 文件名或路径。
        dataset_dir: 数据集目录。
        nrows: 只读取前 N 行（便于快速实验与小样本自检）。
        usecols: 只读取指定列；``None`` 表示读取全部。
        validate: 是否校验列名与顺序（子集读取时自动跳过）。
        parse_time: 是否把 ``Timestamp`` 解析为时间类型。

    Returns:
        DataFrame，并附加 ``source_file`` 列。

    Raises:
        ImportError: 未安装 pandas。
        FileNotFoundError: 文件不存在。
        ValueError: 列名与 schema 不一致，或时间戳无法解析。
    """
    pd = _require_pandas()
    path = resolve_path(name, dataset_dir)

    frame = pd.read_csv(path, nrows=nrows, usecols=list(usecols) if usecols else None)

    if validate and usecols is None and nrows != 0:
        schema.assert_columns(list(frame.columns))

    if parse_time and schema.TIME_COLUMN in frame.columns:
        frame = standardize_timestamp(frame)

    frame[SOURCE_COLUMN] = path.name
    return frame


def iter_features(
    name: str,
    dataset_dir: str | Path = schema.DATASET_DIR,
    chunksize: int = DEFAULT_CHUNKSIZE,
    parse_time: bool = True,
) -> Iterator[Any]:
    """分块迭代读取大文件（1.6M 行级别的数据不建议整表载入）。

    Args:
        name: 文件名或路径。
        dataset_dir: 数据集目录。
        chunksize: 每块行数。
        parse_time: 是否解析时间列。

    Yields:
        每个数据块的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 每块的列名与 schema 不一致。
    """
    pd = _require_pandas()
    path = resolve_path(name, dataset_dir)

    for chunk in pd.read_csv(path, chunksize=chunksize):
        schema.assert_columns(list(chunk.columns))
        if parse_time:
            chunk = standardize_timestamp(chunk)
        chunk[SOURCE_COLUMN] = path.name
        yield chunk


def load_normal_dataset(
    dataset_dir: str | Path = schema.DATASET_DIR,
    nrows_per_file: int | None = None,
    files: Sequence[str] = schema.NORMAL_FILES,
) -> Any:
    """读取全部正常数据并纵向拼接（§5.2 的 9 月 12–30 日）。

    Args:
        dataset_dir: 数据集目录。
        nrows_per_file: 每个文件读取的行数上限；``None`` 表示全部。
        files: 文件清单。

    Returns:
        拼接后的 DataFrame（``ignore_index=True``）。

    Raises:
        ImportError: 未安装 pandas。
    """
    pd = _require_pandas()
    frames = [
        load_features_csv(name, dataset_dir=dataset_dir, nrows=nrows_per_file)
        for name in files
    ]
    return pd.concat(frames, ignore_index=True)


def load_three_state_dataset(
    dataset_dir: str | Path = schema.DATASET_DIR,
    nrows: int | None = None,
) -> Any:
    """读取三态（Normal / Spoofing / Jamming）数据（§5.2 的 1221 数据）。

    Args:
        dataset_dir: 数据集目录。
        nrows: 读取行数上限。

    Returns:
        DataFrame。

    Raises:
        ImportError: 未安装 pandas。
    """
    return load_features_csv(schema.THREE_STATE_FILE, dataset_dir=dataset_dir, nrows=nrows)


def assign_split(frame: Any, day_column: str = schema.DAY_COLUMN) -> Any:
    """为每行标注所属划分（§16.2）。

    Args:
        frame: 正常数据表。
        day_column: 日期列名。

    Returns:
        含 ``split`` 列的新表（取值为 ``train`` / ``validation`` / ``holdout`` / ``unknown``）。

    Raises:
        ImportError: 未安装 pandas。
    """
    pd = _require_pandas()

    def _classify(value: Any) -> str:
        try:
            return schema.split_of_day(value)
        except (ValueError, TypeError):
            return "unknown"

    classified = frame.copy()
    unique_days = pd.unique(classified[day_column])
    mapping = {day: _classify(day) for day in unique_days}
    classified["split"] = classified[day_column].map(mapping)
    return classified


def filter_split(
    frame: Any,
    name: str,
    day_column: str = schema.DAY_COLUMN,
) -> Any:
    """按划分名称过滤数据。

    Args:
        frame: 数据表。
        name: ``"train"`` / ``"validation"`` / ``"holdout"``。
        day_column: 日期列名。

    Returns:
        该划分的数据子集（索引已重置）。

    Raises:
        KeyError: 划分名不存在。
        ValueError: 该划分没有匹配到任何行。
    """
    days = schema.days_for_split(name)
    mask = frame[day_column].isin(list(days))
    subset = frame.loc[mask].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"划分 {name!r}（日期 {days}）未匹配到任何行："
            f"请检查 {day_column} 列取值是否与 schema 一致"
        )
    return subset


def describe_dataset(dataset_dir: str | Path = schema.DATASET_DIR) -> Mapping[str, Any]:
    """汇总数据集文件的存在性与大小，便于环境自检。

    Args:
        dataset_dir: 数据集目录。

    Returns:
        文件名 → ``{"exists": bool, "size_mb": float, "expected_rows": int}``。
    """
    report: dict[str, Any] = {}
    for name in (*schema.NORMAL_FILES, schema.THREE_STATE_FILE):
        path = Path(dataset_dir) / name
        report[name] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1) if path.exists() else 0.0,
            "expected_rows": schema.EXPECTED_ROWS.get(name),
        }
    return report
