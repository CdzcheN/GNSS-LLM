"""训练数据划分、窗口配置与防泄漏校验。

对应开发文档
    §6.2 时间窗口（默认 W = 60 s；敏感性实验 10 / 30 / 60 / 120 s）、
    §16.2 数据划分（正常数据 12–13 → Train、14 → Validation、15 → Hold-out；
    1221 优先按攻击事件划分；Leave-One-Event-Out：19 个欺骗 / 10 个干扰事件区间）、
    §16.3 防泄漏规范（禁止随机窗口打散、测试集参与 scaler fit 或特征选择）、
    §2.5 时间序列禁止随机泄漏。

职责
    1. 描述划分协议（时间段划分与事件级划分）；
    2. 提供可执行的防泄漏校验：划分不得重叠、窗口不得跨划分边界取未来信息；
    3. 定义窗口配置并校验其合法性（与 §6.2 的敏感性集合对齐）。

不做（边界）
    - 不做磁盘 I/O 与解析（属 M1，scripts/extract_features.py）；
    - 不计算标准化统计量（scaler 只能来自训练集，§6.4）；
    - 不做任何随机划分（§16.3 明确禁止）。

输入 / 输出
    输入：划分定义（时间区间 / 事件 id）与窗口配置
    输出：校验结果；或窗口张量（B × W × D）的构建入口

关键约束
    - 划分的重叠检查必须是硬失败（抛异常），不得只打印警告；
    - 全部校验为纯函数，不依赖 pandas/numpy，便于在数据接入前先行验证协议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.data import schema

#: §6.2 的窗口敏感性实验取值（秒）。
WINDOW_CHOICES: tuple[int, ...] = (10, 30, 60, 120)

#: 默认窗口长度（§6.2：W = 60 s）。
DEFAULT_WINDOW_S: int = 60

#: §16.2 正常数据实验的划分（日期）。
#:
#: 权威定义在 ``src/data/schema.py:SPLIT_BY_DAY``（当前数据集为 9 月 12–30 日共 19 天），
#: 此处直接引用以免两处独立演化。
NORMAL_SPLITS: Mapping[str, tuple[int, ...]] = dict(schema.SPLIT_BY_DAY)

#: §16.2 的 Leave-One-Event-Out 事件区间数量。
LOEO_EVENT_COUNTS: Mapping[str, int] = {"spoofing": 19, "jamming": 10}


def expand_day_range(value: Sequence[Any]) -> tuple[int, ...]:
    """把配置中的日期写法展开为日期元组。

    支持两种等价写法：

    - 区间：``[12, 20]`` → ``(12, 13, …, 20)``（含端点）；
    - 枚举：``[12, 13, 14]`` → ``(12, 13, 14)``。

    Args:
        value: 配置中的取值（整数序列）。

    Returns:
        日期元组。

    Raises:
        ValueError: 取值为空或无法转换为整数。
    """
    try:
        items = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"日期取值需为整数序列，实际为 {value!r}") from exc
    if not items:
        raise ValueError("日期取值不能为空")
    if len(items) == 2 and items[0] < items[1]:
        return tuple(range(items[0], items[1] + 1))
    return tuple(items)


@dataclass(slots=True)
class WindowConfig:
    """滑动窗口配置（§6.2）。

    Attributes:
        window_s: 窗口长度（秒）。
        stride_s: 滑动步长（秒）；``None`` 表示等于窗口长度（不重叠）。
        allow_sensitivity: 是否允许非默认窗口（用于 §18.7 敏感性实验）。
    """

    window_s: int = DEFAULT_WINDOW_S
    stride_s: int | None = None
    allow_sensitivity: bool = True

    def __post_init__(self) -> None:
        """校验窗口取值。"""
        if self.window_s <= 0:
            raise ValueError(f"window_s 必须为正：{self.window_s}")
        if self.stride_s is not None and self.stride_s <= 0:
            raise ValueError(f"stride_s 必须为正：{self.stride_s}")
        if not self.allow_sensitivity and self.window_s != DEFAULT_WINDOW_S:
            raise ValueError(f"未开启敏感性实验时窗口只能取 {DEFAULT_WINDOW_S}s（§6.2）")
        if self.allow_sensitivity and self.window_s not in WINDOW_CHOICES:
            raise ValueError(f"窗口取值需属于 §6.2 的敏感性集合 {WINDOW_CHOICES}，实际为 {self.window_s}")

    @property
    def effective_stride_s(self) -> int:
        """实际步长（未指定时等于窗口长度）。"""
        return self.window_s if self.stride_s is None else self.stride_s


@dataclass(slots=True)
class SplitSpec:
    """数据划分定义（§16.2）。

    时间段划分与事件级划分二选一：

    Attributes:
        name: 划分名称（如 ``normal`` / ``event_1221``）。
        strategy: ``"time"`` 或 ``"leave_one_event_out"``。
        time_ranges: 时间段划分，形如 ``{"train": [12, 13], ...}``。
        event_ids: 事件级划分，形如 ``{"train": [...], "holdout": [...]}``。
        notes: 备注（例如“一期基线可用前 70% / 后 30%”）。
    """

    name: str
    strategy: str = "time"
    time_ranges: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    event_ids: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        """校验划分策略与内容一致性。"""
        if self.strategy not in ("time", "leave_one_event_out"):
            raise ValueError(f"未知划分策略 {self.strategy!r}")
        if self.strategy == "time" and not self.time_ranges:
            raise ValueError("time 划分必须给出 time_ranges")
        if self.strategy == "leave_one_event_out" and not self.event_ids:
            raise ValueError("leave_one_event_out 划分必须给出 event_ids")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SplitSpec":
        """由 config.yaml 的 ``splits`` 段构造。

        Args:
            config: 含 ``normal`` 与 ``event_1221`` 的映射。

        Returns:
            SplitSpec。

        Note:
            ``event_1221.strategy`` 为 ``leave_one_event_out`` 时，事件清单尚未固化
            （§16.2 只给出区间数量），因此 ``event_ids`` 允许为空并在后续补全。
        """
        normal = config.get("normal") or {}
        time_ranges = {
            key: expand_day_range(value)
            for key, value in normal.items()
            if key != "notes"
        }
        return cls(
            name="event_1221" if "event_1221" in config else "normal",
            strategy="time",
            time_ranges=time_ranges or dict(NORMAL_SPLITS),
        )


def assert_disjoint(named_sets: Mapping[str, Iterable[Any]]) -> None:
    """校验各划分互不相交（§16.3 防泄漏的硬性检查）。

    Args:
        named_sets: 划分名 → 元素集合（时间窗标识或事件 id）。

    Raises:
        ValueError: 存在跨划分重叠，消息中给出重叠元素。
    """
    materialized = {name: set(values) for name, values in named_sets.items()}
    names = sorted(materialized)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = materialized[left] & materialized[right]
            if overlap:
                raise ValueError(
                    f"划分 {left!r} 与 {right!r} 存在重叠元素 {sorted(overlap)[:5]}"
                    "（违反 §16.3 防泄漏规范）"
                )


def assert_no_shuffle(seed_used_for_split: str | None) -> None:
    """拒绝以随机方式划分数据集（§2.5、§16.3）。

    Args:
        seed_used_for_split: 若划分过程使用了随机种子，则传入该种子的说明文本。

    Raises:
        ValueError: 一旦说明划分使用了随机打散。
    """
    if seed_used_for_split:
        raise ValueError(
            "禁止对连续 GNSS 时间序列做随机打散划分（§2.5、§16.3）；"
            f"收到随机划分说明：{seed_used_for_split!r}"
        )


def build_windows(
    frame: Any,
    feature_columns: Sequence[str],
    label_column: str = "Label",
    window_s: int = DEFAULT_WINDOW_S,
    stride_s: int | None = None,
) -> tuple[Any, Any, Any]:
    """构建 B × W × D 窗口张量（§6.1 流水线末端、§6.2 时间窗口）。

    Args:
        frame: 已按时间排序的特征表（含特征列与标签列）。
        feature_columns: 参与建模的特征列（决定 D 维）。
        label_column: 标签列名（§1.3 三态）。
        window_s: 窗口长度（行数；1 Hz 数据下等于秒数）。
        stride_s: 滑动步长；``None`` 表示等于窗口长度（不重叠）。

    Returns:
        ``(X, y, end_indices)``：

        - ``X``：形状 ``(B, W, D)`` 的 float64 数组；
        - ``y``：形状 ``(B,)`` 的标签（取**窗口末行**标签，保证因果性）；
        - ``end_indices``：每个窗口末行在 ``frame`` 中的位置，便于回溯时间戳。

    Raises:
        ImportError: 未安装 numpy。
        ValueError: 列缺失、窗口非法，或数据不足一个窗口。

    Note:
        标签取窗口末行而非窗口内多数投票：前者不会让窗口“提前知道”未来标签，
        与在线检测的真实时序一致（§2.5、§16.3 防泄漏）。
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "构建窗口张量需要 numpy，请先执行 `pip install -r requirements.txt`"
        ) from exc

    if window_s < 1:
        raise ValueError(f"window_s 必须 >= 1：{window_s}")
    stride = window_s if stride_s is None else int(stride_s)
    if stride < 1:
        raise ValueError(f"stride_s 必须 >= 1：{stride}")

    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少特征列：{missing[:5]}（共 {len(missing)}）")
    if label_column not in frame.columns:
        raise ValueError(f"缺少标签列 {label_column!r}")

    values = frame[list(feature_columns)].to_numpy(dtype="float64")
    labels = frame[label_column].to_numpy()
    if len(values) < window_s:
        raise ValueError(
            f"数据行数 {len(values)} 少于一个窗口 {window_s}（§6.2），无法构建窗口"
        )

    windows = []
    targets = []
    ends = []
    for end in range(window_s - 1, len(values), stride):
        start = end - window_s + 1
        windows.append(values[start : end + 1])
        targets.append(labels[end])
        ends.append(end)

    return (
        np.stack(windows),
        np.asarray(targets),
        np.asarray(ends, dtype="int64"),
    )


@dataclass(slots=True)
class EventSplit:
    """事件级数据划分（§16.2）。

    每条划分由若干**连续行区间**组成，而非单一行集合——因为时间序列在拼接不连续段时
    会产生“假窗口”，必须按段分别构建窗口（见 ``windows_from_segments``）。

    Attributes:
        train_segments: 训练侧的行区间（``(start, end)`` 闭区间）。
        validation_segments: 验证侧的行区间。
        details: 每类别的事件段统计（段数 / 训练段数 / 验证段数）。
    """

    train_segments: tuple[tuple[int, int], ...] = ()
    validation_segments: tuple[tuple[int, int], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def indices(self, part: str = "train") -> Any:
        """把段展开为行索引数组。

        Args:
            part: ``"train"`` 或 ``"validation"``。

        Returns:
            行索引的 numpy 数组。

        Raises:
            ValueError: ``part`` 非法。
        """
        import numpy as np

        if part == "train":
            segments = self.train_segments
        elif part == "validation":
            segments = self.validation_segments
        else:
            raise ValueError(f"part 需为 train/validation，实际为 {part!r}")

        blocks = [np.arange(start, end + 1) for start, end in segments]
        if not blocks:
            return np.array([], dtype="int64")
        return np.concatenate(blocks)

    @property
    def train_rows(self) -> int:
        """训练侧行数。"""
        return int(sum(end - start + 1 for start, end in self.train_segments))

    @property
    def validation_rows(self) -> int:
        """验证侧行数。"""
        return int(sum(end - start + 1 for start, end in self.validation_segments))

    def summary(self) -> Mapping[str, Any]:
        """返回可写入实验记录的摘要。"""
        return {
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "train_segments": len(self.train_segments),
            "validation_segments": len(self.validation_segments),
            "per_class": dict(self.details),
        }


def find_event_segments(
    frame: Any,
    label_column: str = "Label",
) -> Mapping[int, tuple[tuple[int, int], ...]]:
    """识别连续同标签的段（§16.2 的事件区间）。

    Args:
        frame: 已按时间排序的特征表。
        label_column: 标签列。

    Returns:
        ``{标签: ((start, end), ...)}``，Normal（0）与各攻击类别分别给出。

    Raises:
        ValueError: 缺少标签列。
    """
    if label_column not in frame.columns:
        raise ValueError(f"缺少标签列 {label_column!r}")

    labels = frame[label_column].to_numpy()
    if len(labels) == 0:
        return {}

    segments: dict[int, list[tuple[int, int]]] = {}
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.setdefault(int(labels[start]), []).append((start, index - 1))
            start = index
    return {label: tuple(items) for label, items in segments.items()}


def split_by_events(
    frame: Any,
    label_column: str = "Label",
    validation_event_ratio: float = 1.0 / 3.0,
    normal_validation_ratio: float = 0.3,
) -> EventSplit:
    """按事件划分数据，**保证验证集含所有出现过的类别**（§16.2）。

    动机：1221 当天三类样本在时间上并不交错（Spoofing 12:32–16:44、Jamming 16:56–17:20、
    Normal 全天）。若按时间 70/30 切分，异常会全部落进训练集、验证集只剩 Normal，
    于是“全判正常”在验证集上看起来完美，早停会挑出无效模型。

    做法：把每类的连续段按时间顺序排列，**末尾若干段**归验证集；
    每类至少保留 1 段给训练集，且只有 1 段的类别不做切分（整体归训练）。

    Args:
        frame: 已按时间排序的特征表。
        label_column: 标签列。
        validation_event_ratio: 异常类别用于验证的事件段比例。
        normal_validation_ratio: Normal 段用于验证的比例。

    Returns:
        EventSplit。

    Raises:
        ValueError: 缺少标签列，或切分后任一侧为空。
    """
    segments = find_event_segments(frame, label_column)
    if not segments:
        raise ValueError("未识别到任何数据段")

    train_segments: list[tuple[int, int]] = []
    validation_segments: list[tuple[int, int]] = []
    details: dict[str, Any] = {}

    for label, segs in sorted(segments.items()):
        ratio = normal_validation_ratio if label == 0 else validation_event_ratio
        count = len(segs)
        if count <= 1:
            holdout = 0
        else:
            holdout = int(round(count * ratio))
            holdout = max(1, min(holdout, count - 1))  # 至少留 1 段做验证、1 段做训练

        split_at = count - holdout
        train_segments.extend(segs[:split_at])
        validation_segments.extend(segs[split_at:])
        details[schema.LABEL_NAMES.get(label, str(label))] = {
            "segments": count,
            "train": split_at,
            "validation": holdout,
        }

    split = EventSplit(
        train_segments=tuple(sorted(train_segments)),
        validation_segments=tuple(sorted(validation_segments)),
        details=details,
    )
    if not split.train_segments:
        raise ValueError("切分后训练侧为空，请检查数据或调低验证比例")
    if not split.validation_segments:
        raise ValueError("切分后验证侧为空，请检查数据或调高验证比例")
    return split


def windows_from_segments(
    frame: Any,
    segments: Sequence[tuple[int, int]],
    feature_columns: Sequence[str],
    label_column: str = "Label",
    window_s: int = DEFAULT_WINDOW_S,
    stride_s: int | None = None,
) -> tuple[Any, Any]:
    """在给定行区间内**分别**构建窗口（避免跨段拼接产生假窗口）。

    时间序列被切成多段后直接拼接会产生“上一段尾 + 下一段头”的伪窗口，
    这些窗口在真实时间上并不连续，必须避免（§2.5、§16.3）。

    Args:
        frame: 特征表（已排序）。
        segments: 行区间序列。
        feature_columns: 特征列。
        label_column: 标签列。
        window_s: 窗口长度。
        stride_s: 步长；``None`` 表示等于窗口长度。

    Returns:
        ``(X, y)``：形状 ``(B, W, D)`` 与 ``(B,)``。

    Raises:
        ImportError: 未安装 numpy。
        ValueError: 所有段都短于一个窗口。
    """
    import numpy as np

    windows: list[Any] = []
    targets: list[Any] = []
    for start, end in segments:
        block = frame.iloc[start : end + 1]
        if len(block) < window_s:
            continue
        block_x, block_y, _ = build_windows(
            block, feature_columns, label_column, window_s=window_s, stride_s=stride_s
        )
        windows.append(block_x)
        targets.append(block_y)

    if not windows:
        raise ValueError(
            f"所有段都短于一个窗口（window_s={window_s}），无法构建样本；"
            "请调小窗口或改用更长的连续段"
        )
    return np.concatenate(windows), np.concatenate(targets)


def loeo_folds(counts: Mapping[str, int] = LOEO_EVENT_COUNTS) -> Mapping[str, int]:
    """返回 Leave-One-Event-Out 的折数（§16.2）。

    Args:
        counts: 各攻击类型的事件区间数量。

    Returns:
        形如 ``{"spoofing": 19, "jamming": 10}`` 的映射。

    Note:
        留一事件的**具体区间清单**尚未固化（§16.2 仅给出数量），
        因此本函数只暴露规模，供实验计划与记录使用。
    """
    return dict(counts)
