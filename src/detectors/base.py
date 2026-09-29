"""检测器统一接口与结构化结果定义（检测策略库 M4 的基础契约）。

对应开发文档
    §1.3 核心任务（GNSS 三态识别）、§8.1 统一接口、§8.4 Detector Registry、
    §12.1 结果标准化、§2.2 物理可解释优先、§2.4 大模型不参与核心判定。

职责
    1. 定义三态标签 AttackType（0=Normal / 1=Spoofing / 2=Jamming）；
    2. 定义所有检测器统一返回的 DetectionResult，字段与 §8.1 一一对应；
    3. 定义检测器基类 BaseDetector，强制统一签名 run(data, context, config)；
    4. 定义注册表所需元信息 DetectorMeta（§8.4 的 8 个字段）。

不做（边界）
    - 不实现任何具体检测算法（S1–S7 各自实现）；
    - 不改写检测标签或置信度（§2.4：编排层与 LLM 均无权修改检测器输出）；
    - 不读写文件、不访问网络、不依赖第三方库（保证可被任意模块安全导入）。

输入 / 输出
    输入：data（滑动窗口 / 特征表）、context（GNSS Context）、config（配置映射）
    输出：DetectionResult

关键约束
    - 所有检测器必须实现同一签名，智能体只面向 §8.4 的“可调用工具”；
    - run() 必须可复现：同输入同配置给出相同结果（§20.4 固定随机种子）；
    - evidence 必须是可追溯的结构化字段，不允许只给结论（§2.2）。

待扩展
    - 批量/流式执行可在子类中另行提供 run_batch()，基类接口保持单窗口语义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping


class AttackType(IntEnum):
    """GNSS 三态标签（§1.3）。

    数值与 §5.2 数据集中的 Label 列一致，可直接用于监督学习。
    """

    NORMAL = 0
    SPOOFING = 1
    JAMMING = 2


class DetectorStatus(str, Enum):
    """检测器执行状态（§8.1 的 status 字段）。"""

    OK = "ok"
    DEGRADED = "degraded"  # 数据质量不足，仍给出降级结论（§5.4 数据质量问题）
    SKIPPED = "skipped"    # 因上下文或预算未被执行（§9.5 原则 A 低成本优先）
    ERROR = "error"        # 执行异常，需由融合层按缺失处理（§12.4 冲突处理）


@dataclass(slots=True)
class DetectionResult:
    """检测器统一结构化输出（字段与 §8.1 完全一致）。

    Attributes:
        detector_id: 产生该结果的检测器标识（§8.4 detector_id）。
        attack_type: 三态判定结果。
        confidence: 置信度，取值 [0, 1]，用于 §12.2 置信度加权融合。
        evidence: 结构化检测证据，键值必须可追溯（§2.2）。
        timestamp: 该结果对应的窗口时间戳。
        latency_ms: 本次检测耗时（毫秒），用于 §17.4 策略层指标。
        data_quality: 输入数据质量评分，用于 §12.2 权重调整。
        status: 执行状态。
    """

    detector_id: str
    attack_type: AttackType
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    latency_ms: float | None = None
    data_quality: float | None = None
    status: DetectorStatus = DetectorStatus.OK

    def __post_init__(self) -> None:
        """校验取值范围，避免非法置信度污染下游融合（§12.2）。"""
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence 必须在 [0, 1] 内，实际为 {self.confidence!r}")

    @property
    def is_normal(self) -> bool:
        """是否为正常判定。"""
        return self.attack_type is AttackType.NORMAL

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典，供事件记录（§13.1）与 LLM 输入（§14.2）使用。"""
        return {
            "detector_id": self.detector_id,
            "attack_type": int(self.attack_type),
            "attack_name": self.attack_type.name,
            "confidence": float(self.confidence),
            "evidence": dict(self.evidence),
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "data_quality": self.data_quality,
            "status": self.status.value,
        }


@dataclass(slots=True)
class DetectorMeta:
    """注册表元信息（字段与 §8.4 Detector Registry 一致）。

    Attributes:
        detector_id: 唯一标识，注册表主键。
        input_schema: 期望输入结构描述。
        output_schema: 输出结构描述（通常为 DetectionResult）。
        supported_context: 适用的上下文分量（§7.2 的 S/Q/O/N/H/D 子集）。
        expected_latency_ms: 预期时延，用于 §9.5 原则 A 与 §17.4 策略层指标。
        expected_cost: 预期计算开销，用于低开销优先的编排。
        reliability: 历史可靠度（0–1），由 §17.4 指标回填。
        version: 检测器版本，便于实验复现（§20.3 model_version）。
    """

    detector_id: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    supported_context: tuple[str, ...] = ()
    expected_latency_ms: float | None = None
    expected_cost: float | None = None
    reliability: float = 1.0
    version: str = "v0"


def numeric(
    data: Mapping[str, Any] | None,
    key: str,
    default: float | None = None,
) -> float | None:
    """从特征映射中安全取实数。

    用于检测器读取特征值：排除缺失键、``None``、NaN 与布尔值，
    避免把“无数据”当成“数值 0”（§5.4 Q3 的同类问题）。

    Args:
        data: 特征映射（通常为某一行特征）。
        key: 特征名。
        default: 取不到时的返回值。

    Returns:
        浮点值或 ``default``。
    """
    if data is None:
        return default
    try:
        value = data[key]
    except (KeyError, IndexError, TypeError):
        return default
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number  # NaN 检查


class BaseDetector(ABC):
    """检测器基类：统一接口 + 注册元信息（§8.1、§8.4、§2.1 检测与决策解耦）。

    子类只需实现 ``detector_id``、``supported_context`` 与 ``run()``。
    """

    #: 唯一标识，同时作为注册表键（§8.4 detector_id）。
    detector_id: str = "base"
    #: 该检测器适用的上下文分量（§7.2 的 S/Q/O/N/H/D）。
    supported_context: tuple[str, ...] = ()
    #: 版本号，参与实验记录（§20.3）。
    version: str = "v0"

    @abstractmethod
    def run(self, data: Any, context: Any, config: Mapping[str, Any]) -> DetectionResult:
        """执行检测并返回结构化结果（§8.1）。

        Args:
            data: 当前滑动窗口数据或特征表（§6.2 时间窗口）。
            context: 当前 GNSS Context（§7 章）。
            config: 该次调用的配置映射（阈值、权重等）。

        Returns:
            DetectionResult: 统一结构化检测结果。

        Raises:
            NotImplementedError: 子类尚未实现时由子类抛出。
        """
        raise NotImplementedError

    def describe(self) -> DetectorMeta:
        """返回注册表元信息，供 §8.4 / §9.6 的工具注册使用。"""
        return DetectorMeta(
            detector_id=self.detector_id,
            supported_context=tuple(self.supported_context),
            version=self.version,
        )

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"<{type(self).__name__} id={self.detector_id} version={self.version}>"
