"""M3 GNSS 上下文状态建模：把多源特征压缩为可供策略选择使用的结构化 Context。

对应开发文档
    §7 章（7.1 模块定位、7.2 Context 组成、7.3 上下文特征、7.4 上下文输出）、
    §9.2 决策输入、§18.1 Context 消融实验、§2.3 动态选择必须可验证。

职责
    1. 定义 Context 结构与六分量 ``C_t = [S_t, Q_t, O_t, N_t, H_t, D_t]``（§7.2）；
    2. 把 M2 的特征表按列语义分派到六个分量（信号 / 质量 / 观测 / 导航 / 历史 / 检测器）；
    3. 估计上下文置信度并透传数据质量（§7.4）。

不做（边界）
    - 不直接让智能体读取原始特征大矩阵（§7.1：先形成结构化 Context）；
    - 不做策略选择或检测器调用（属 M5，§9）；
    - 不做任何标签判定（§2.4）。

输入 / 输出
    输入：M2 特征表（一行 = 一个历元）与可选的数据质量评分
    输出：Context 实例（可序列化，用于 §20.3 实验记录）

关键约束
    - 六分量必须齐备；缺项用空映射显式表达，禁止静默填充（§2.2）；
    - 历史分量 H_t 只能引用已发生的行（本实现不含未来信息，§2.5、§16.3）；
    - 列 → 分量的分派规则必须可解释且可复算（下方 ``_component_of``）；
    - ``enabled=False`` 时返回 ``None``，作为 §18.1 “无 Context”对照组的开关。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

#: 六分量名称（§7.2），顺序与 C_t 定义一致。
CONTEXT_COMPONENTS: tuple[str, ...] = ("S", "Q", "O", "N", "H", "D")

#: 各分量的说明（写入实验记录，便于人工复核）。
COMPONENT_MEANING: Mapping[str, str] = {
    "S": "Signal Context：C/N0 与可见星数等信号状态",
    "Q": "Quality Context：有效卫星数、缺口与数据质量",
    "O": "Observation Context：伪距残差统计与超限情况",
    "N": "Navigation Context：PVT 精度、DOP 与钟差差分",
    "H": "Historical Context：最近若干历元的状态回顾",
    "D": "Detector Context：各检测器的历史表现与开销",
}

# 列语义分派规则（§7.3）：按列名前缀/名称归类到分量。
_SIGNAL_PREFIXES: tuple[str, ...] = ("cn0_", "sat_count_", "avg_cno_")
_OBSERVATION_PREFIXES: tuple[str, ...] = ("res_",)
_NAVIGATION_PREFIXES: tuple[str, ...] = (
    "hAcc", "vAcc", "tAcc", "gSpeed", "pDOP", "tDOP", "hDOP", "clkB", "clkD",
)
_QUALITY_COLUMNS: tuple[str, ...] = (
    "valid_sat_count", "sat_mask", "miss_mask", "gap_before_s", "data_quality",
)
#: 历史分量默认保留的历元数（§7.3 历史类特征）。
DEFAULT_HISTORY_WINDOWS: int = 5


@dataclass(slots=True)
class Context:
    """GNSS 上下文状态 ``C_t = [S_t, Q_t, O_t, N_t, H_t, D_t]``（§7.2）。

    Attributes:
        S: 信号/频谱状态。
        Q: C/N0 与数据完整性。
        O: 观测一致性与残差。
        N: PVT/DOP。
        H: 最近若干历元的历史状态。
        D: 各检测器历史表现。
        confidence: 上下文置信度（§7.4）。
        data_quality: 数据质量评分（取值 [0, 1]）。
        timestamp: 对应窗口时间。
    """

    S: Mapping[str, float] = field(default_factory=dict)
    Q: Mapping[str, float] = field(default_factory=dict)
    O: Mapping[str, float] = field(default_factory=dict)
    N: Mapping[str, float] = field(default_factory=dict)
    H: Mapping[str, float] = field(default_factory=dict)
    D: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    data_quality: float | None = None
    timestamp: str | None = None
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验质量与置信度取值范围（避免越界值进入融合权重，§12.2）。"""
        for name in ("confidence", "data_quality"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 内，实际为 {value!r}")

    def component(self, name: str) -> Mapping[str, float]:
        """按名称取分量（§7.2）。

        Args:
            name: ``"S" | "Q" | "O" | "N" | "H" | "D"``。

        Returns:
            对应分量的特征映射。

        Raises:
            KeyError: 名称不在六分量内。
        """
        if name not in CONTEXT_COMPONENTS:
            raise KeyError(f"未知 Context 分量：{name!r}，可选 {CONTEXT_COMPONENTS}")
        return getattr(self, name)

    @property
    def is_complete(self) -> bool:
        """六个分量是否全部非空（含运行期注入的 D，§7.2）。"""
        return all(self.component(name) for name in CONTEXT_COMPONENTS)

    @property
    def is_ready(self) -> bool:
        """除 ``D``（检测器历史，由运行期注入）外的五个分量是否非空。

        离线编码时 D 必然为空，若用 ``is_complete`` 判断会得到恒假的结果；
        ``is_ready`` 才是“离线可用的上下文”判据。
        """
        return all(self.component(name) for name in CONTEXT_COMPONENTS if name != "D")

    @property
    def missing_components(self) -> tuple[str, ...]:
        """列出为空的分量，便于日志与排查。"""
        return tuple(name for name in CONTEXT_COMPONENTS if not self.component(name))

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化字典。"""
        return asdict(self)

    def flattened(self) -> dict[str, float]:
        """把六分量展平为 ``"S.cn0_delta_db"`` 形式的单层映射。

        便于写入扁平结构（如事件证据、CSV 特征列）。

        Returns:
            展平后的映射。
        """
        flat: dict[str, float] = {}
        for name in CONTEXT_COMPONENTS:
            for key, value in self.component(name).items():
                flat[f"{name}.{key}"] = value
        return flat


@dataclass(slots=True)
class FlagThresholds:
    """环境状态判据的阈值（用于导出 §8.3 的 flags）。

    这些阈值与检测器判据**同源**（都取自 ``config.yaml`` 的 ``detectors`` 段），
    以保证“策略路由依据”与“检测器判据”不会各说各话（§2.3 可验证）。

    Attributes:
        cn0_drop_db: C/N0 下降阈值（dB）。
        sat_count_drop: 卫星数下降阈值。
        residual_outlier_count: 残差超限个数阈值。
        acc_rise: 精度指标相对基线抬升阈值。
        dop_rise: DOP 相对基线抬升阈值。
    """

    cn0_drop_db: float = -3.0
    sat_count_drop: float = -3.0
    residual_outlier_count: float = 2.0
    acc_rise: float = 20.0
    dop_rise: float = 1.0

    @classmethod
    def from_config(cls, detectors_config: Mapping[str, Any] | None) -> "FlagThresholds":
        """由 ``config.yaml`` 的 ``detectors`` 段构造。

        Args:
            detectors_config: ``detectors`` 段映射（可含各检测器子段）。

        Returns:
            FlagThresholds（缺项保留默认值）。
        """
        section = dict(detectors_config or {})
        cno = dict(section.get("cno") or {})
        satellite = dict(section.get("satellite") or {})
        observation = dict(section.get("observation") or {})
        pvt = dict(section.get("pvt") or {})
        # slots=True 时类属性是 slot 描述符，必须通过实例读取字段默认值
        defaults = cls()
        return cls(
            cn0_drop_db=float(cno.get("cn0_drop_db", defaults.cn0_drop_db)),
            sat_count_drop=float(satellite.get("sat_count_drop", defaults.sat_count_drop)),
            residual_outlier_count=float(
                observation.get("outlier_count", defaults.residual_outlier_count)
            ),
            acc_rise=float(pvt.get("acc_rise", defaults.acc_rise)),
            dop_rise=float(pvt.get("dop_rise", defaults.dop_rise)),
        )


def derive_flags(
    data: Mapping[str, Any] | None,
    thresholds: FlagThresholds,
) -> tuple[str, ...]:
    """由特征值导出环境状态判据（键名与 §8.3 的策略适用性矩阵一致）。

    Args:
        data: 某历元的特征映射（来自 ``src/features/*`` 的派生列）。
        thresholds: 判据阈值。

    Returns:
        命中的环境状态键元组，可能包含：

        - ``cn0_decreasing``：C/N0 整体下降；
        - ``satellite_abnormal``：卫星数异常；
        - ``observation_inconsistent``：观测量不一致；
        - ``pvt_abnormal``：PVT/DOP 异常。

    Note:
        ``spectrum_abnormal``（模态 B）与 ``metrics_conflict`` / ``high_uncertainty``
        （需多检测器结果）不在此处产生：前者缺数据，后者由融合层在检测之后判定（§12.4）。
    """
    from src.detectors.base import numeric  # 局部导入，避免模块级循环依赖

    flags: list[str] = []

    cn0_delta = numeric(data, "cn0_delta_db")
    if cn0_delta is not None and cn0_delta <= thresholds.cn0_drop_db:
        flags.append("cn0_decreasing")

    sat_delta = numeric(data, "sat_count_delta")
    if sat_delta is not None and sat_delta <= thresholds.sat_count_drop:
        flags.append("satellite_abnormal")

    outliers = numeric(data, "res_outlier_count")
    if outliers is not None and outliers >= thresholds.residual_outlier_count:
        flags.append("observation_inconsistent")

    pvt_abnormal = False
    for name in ("hAcc", "vAcc", "tAcc"):
        value = numeric(data, f"{name}_vs_baseline")
        if value is not None and value >= thresholds.acc_rise:
            pvt_abnormal = True
            break
    if not pvt_abnormal:
        for name in ("pDOP", "tDOP", "hDOP"):
            value = numeric(data, f"{name}_vs_baseline")
            if value is not None and value >= thresholds.dop_rise:
                pvt_abnormal = True
                break
    if pvt_abnormal:
        flags.append("pvt_abnormal")

    return tuple(flags)


def _component_of(column: str) -> str:
    """判断特征列属于哪个 Context 分量（§7.3）。

    Args:
        column: 列名。

    Returns:
        分量名（``S`` / ``Q`` / ``O`` / ``N``）；无法归类时返回 ``"S"`` 之外的 ``"?"``。

    Note:
        分派规则只依赖列名语义，不做统计推断，保证可解释与可复算（§2.2）。
    """
    if column in _QUALITY_COLUMNS:
        return "Q"
    if column.startswith(_SIGNAL_PREFIXES):
        return "S"
    if column.startswith(_OBSERVATION_PREFIXES):
        return "O"
    if column.startswith(_NAVIGATION_PREFIXES):
        return "N"
    return "?"


class ContextEncoder:
    """把多源特征编码为 Context（§7.1、§7.4）。

    ``enabled=False`` 时直接返回 ``None``，作为 §18.1 Context 消融的对照组。
    """

    def __init__(
        self,
        enabled: bool = True,
        history_windows: int = DEFAULT_HISTORY_WINDOWS,
        min_components: int = 3,
        thresholds: FlagThresholds | None = None,
    ) -> None:
        """初始化编码器。

        Args:
            enabled: 是否启用 Context 编码（§18.1 消融开关）。
            history_windows: 历史分量 H_t 保留的历元数。
            min_components: 估计置信度所需的最少非空分量数。
            thresholds: 环境状态判据阈值（用于导出 §8.3 的 flags）；
                ``None`` 时使用默认值。
        """
        self.enabled = bool(enabled)
        self.history_windows = int(history_windows)
        self.min_components = int(min_components)
        self.thresholds = thresholds or FlagThresholds()

    # ------------------------------------------------------------------ 编码

    def encode(
        self,
        signal_features: Mapping[str, float] | None = None,
        quality_features: Mapping[str, float] | None = None,
        observation_features: Mapping[str, float] | None = None,
        navigation_features: Mapping[str, float] | None = None,
        history: Mapping[str, float] | None = None,
        detector_stats: Mapping[str, float] | None = None,
        data_quality: float | None = None,
        timestamp: str | None = None,
    ) -> Context | None:
        """由六分量特征构造 Context。

        Args:
            signal_features: S_t。
            quality_features: Q_t。
            observation_features: O_t。
            navigation_features: N_t。
            history: H_t。
            detector_stats: D_t。
            data_quality: 数据质量评分（§7.4）。
            timestamp: 窗口时间。

        Returns:
            启用时返回 Context；``enabled=False`` 时返回 ``None``。
        """
        if not self.enabled:
            return None

        context = Context(
            S=dict(signal_features or {}),
            Q=dict(quality_features or {}),
            O=dict(observation_features or {}),
            N=dict(navigation_features or {}),
            H=dict(history or {}),
            D=dict(detector_stats or {}),
            confidence=None,
            data_quality=None if data_quality is None else float(data_quality),
            timestamp=timestamp,
        )
        context.confidence = self.estimate_confidence(context)
        return context

    def encode_from_frame(
        self,
        frame: Any,
        index: int,
        history_windows: int | None = None,
        detector_stats: Mapping[str, float] | None = None,
        exclude_columns: Sequence[str] | None = None,
    ) -> Context | None:
        """从特征表的某一行编码 Context（按列语义分派到六分量）。

        历史分量 H_t 由该行之前的窗口均值构成（**不含当前与未来行**，§16.3）。

        Args:
            frame: 特征表（含掩码与特征列）。
            index: 行位置（``iloc`` 语义）。
            history_windows: 历史窗口长度；``None`` 表示用构造参数。
            detector_stats: D_t（外部注入，初期可为空）。
            exclude_columns: 不参与编码的列（如时间列、原始 32 列观测）。

        Returns:
            启用时返回 Context；``enabled=False`` 时返回 ``None``。

        Raises:
            IndexError: ``index`` 越界。
            ImportError: 未安装 pandas。
        """
        if not self.enabled:
            return None

        pd = _require_pandas()
        if index < 0 or index >= len(frame):
            raise IndexError(f"index {index} 越界（表长 {len(frame)}）")

        excluded = set(exclude_columns or ())
        row = frame.iloc[index]
        buckets: dict[str, dict[str, float]] = {name: {} for name in CONTEXT_COMPONENTS}

        for column in frame.columns:
            if column in excluded:
                continue
            value = row[column]
            if not _is_number(value):
                continue
            component = _component_of(str(column))
            if component == "?":
                continue
            buckets[component][str(column)] = float(value)
            if str(column) == "data_quality":
                buckets["Q"][str(column)] = float(value)

        window = self.history_windows if history_windows is None else int(history_windows)
        buckets["H"] = self._history_summary(frame, index, window, excluded)
        buckets["D"] = dict(detector_stats or {})

        data_quality = None
        quality_value = row.get("data_quality") if hasattr(row, "get") else None
        if quality_value is not None and _is_number(quality_value):
            data_quality = float(quality_value)

        context = self.encode(
            signal_features=buckets["S"],
            quality_features=buckets["Q"],
            observation_features=buckets["O"],
            navigation_features=buckets["N"],
            history=buckets["H"],
            detector_stats=buckets["D"],
            data_quality=data_quality,
            timestamp=str(row.get("Timestamp")) if hasattr(row, "get") else None,
        )
        if context is not None:
            # 导出 §8.3 的环境状态判据，供策略层做矩阵路由（§9.2 决策输入）
            context.flags = derive_flags(dict(row), self.thresholds)
        return context

    def encode_batch(
        self,
        frame: Any,
        step: int = 1,
        exclude_columns: Sequence[str] | None = None,
    ) -> list[Context | None]:
        """批量编码（用于离线训练与 §18.1 消融实验）。

        Args:
            frame: 特征表。
            step: 采样步长（``1`` 表示逐行编码）。
            exclude_columns: 不参与编码的列。

        Returns:
            与采样行数等长的 Context 列表（``enabled=False`` 时元素均为 ``None``）。

        Raises:
            ValueError: ``step`` 非正。
        """
        if step < 1:
            raise ValueError(f"step 必须 >= 1：{step}")
        return [
            self.encode_from_frame(frame, index, exclude_columns=exclude_columns)
            for index in range(0, len(frame), step)
        ]

    # ------------------------------------------------------------------ 辅助

    def _history_summary(
        self,
        frame: Any,
        index: int,
        window: int,
        excluded: set[str],
    ) -> dict[str, float]:
        """构造历史分量为“前 window 行的均值”（不含当前行）。

        Args:
            frame: 特征表。
            index: 当前行。
            window: 窗口长度。
            excluded: 排除列。

        Returns:
            ``{"hist_mean__<列名>": 均值}`` 形式的映射；无历史行时为空。
        """
        start = max(0, index - window)
        if start >= index:
            return {}

        subset = frame.iloc[start:index]
        summary: dict[str, float] = {}
        for column in subset.columns:
            if column in excluded:
                continue
            if _component_of(str(column)) == "?":
                continue
            series = subset[column]
            if str(series.dtype).startswith("float") or str(series.dtype).startswith("int"):
                summary[f"hist_mean__{column}"] = float(series.mean())
        return summary

    def estimate_confidence(self, context: Context) -> float:
        """估计上下文置信度（§7.4）。

        规则（可复算）：非空分量占比 × 数据质量（质量缺失时按 0.5 计），
        再按“至少 ``min_components`` 个分量非空”给出门限。

        Args:
            context: 待评估的 Context。

        Returns:
            置信度，取值 [0, 1]。
        """
        filled = sum(1 for name in CONTEXT_COMPONENTS if context.component(name))
        coverage = filled / len(CONTEXT_COMPONENTS)
        if filled < self.min_components:
            return 0.0

        quality = 0.5 if context.data_quality is None else float(context.data_quality)
        return max(0.0, min(1.0, coverage * quality * 2.0))


def _require_pandas() -> Any:
    """惰性导入 pandas（未安装时给出可读提示）。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "从特征表编码 Context 需要 pandas，请先执行 `pip install -r requirements.txt`"
        ) from exc
    return pd


def _is_number(value: Any) -> bool:
    """判断值是否为可用于特征计算的实数（排除 NaN 与布尔）。"""
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number  # 排除 NaN


def describe_components() -> Mapping[str, str]:
    """返回六分量含义说明，供报告与文档生成使用。"""
    return dict(COMPONENT_MEANING)
