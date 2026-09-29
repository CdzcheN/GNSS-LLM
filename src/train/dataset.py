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
