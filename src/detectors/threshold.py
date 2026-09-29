"""S1 固定阈值/统计检测器：低成本初筛与回退路径（配置驱动，不硬编码判据）。

对应开发文档
    §8.2 策略分类（S1 固定阈值/统计检测）、§8.3 策略适用性矩阵、
    §9.5 原则 A（低成本优先）、§9.5 原则 D（必须存在回退路径）、§8.1 统一接口。

职责
    1. 按配置给定的阈值规则表评估当前窗口，给出三态判定；
    2. 作为智能体的低成本初筛检测器（§9.4 第一步）；
    3. 作为 Agent 异常时的兜底检测器（§9.5 原则 D、§19.3）。

不做（边界）
    - 不在代码中硬编码任何阈值：判据必须来自 config，便于实验对比与复现（§20.3）；
    - 不做多检测器融合（属 M6，§12）；
    - 不做事件级状态管理（属 M7，§13）。

输入 / 输出
    输入：``data`` 为特征映射（feature → 数值）；``config`` 含 ``rules`` 规则表
    输出：DetectionResult（§8.1）

证据字段（evidence，§8.1 / §2.2）
    - ``matched_rules``：命中的规则描述（含特征名、运算符、阈值、实测值）；
    - ``conflicting_classes``：命中规则指向不同类别时列出冲突类别（§12.4 冲突处理）；
    - ``evaluated_rules``：本次参与评估的规则总数，便于复核。

关键约束
    - 判据必须可追溯到具体特征与阈值，禁止黑盒打分（§2.2）；
    - 无命中规则时判定为 NORMAL，且置信度语义为“未触发任何异常判据”；
    - 执行必须确定性：同输入同配置给出相同结果（§20.4 固定随机种子）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus

#: 允许的比较运算符。保持最小集合，避免配置歧义。
ALLOWED_OPS: tuple[str, ...] = ("<", "<=", ">", ">=")

#: 类别名到枚举的映射，允许配置中用 "jamming" / "spoofing" / "normal" 书写。
_ATTACK_ALIASES: Mapping[str, AttackType] = {
    "normal": AttackType.NORMAL,
    "spoofing": AttackType.SPOOFING,
    "jamming": AttackType.JAMMING,
}


@dataclass(slots=True)
class ThresholdRule:
    """单条阈值规则（判据完全由配置提供）。

    Attributes:
        feature: 参与判别的特征名，需存在于传入的 ``data`` 映射中。
        op: 比较运算符，取值见 ``ALLOWED_OPS``。
        threshold: 阈值。
        attack_type: 命中该规则时指向的类别。
        weight: 该规则的权重，用于计算置信度与解决冲突（§12.4）。
    """

    feature: str
    op: str
    threshold: float
    attack_type: AttackType
    weight: float = 1.0

    def __post_init__(self) -> None:
        """校验运算符与权重合法性。"""
        if self.op not in ALLOWED_OPS:
            raise ValueError(f"不支持的运算符 {self.op!r}，允许：{ALLOWED_OPS}")
        if self.weight < 0:
            raise ValueError(f"weight 不能为负：{self.weight!r}")

    def matches(self, value: float) -> bool:
        """判断给定特征值是否命中该规则。"""
        if self.op == "<":
            return value < self.threshold
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">":
            return value > self.threshold
        return value >= self.threshold

    def describe(self, value: float) -> dict[str, Any]:
        """生成可追溯的规则命中描述（写入 evidence）。"""
        return {
            "feature": self.feature,
            "op": self.op,
            "threshold": self.threshold,
            "value": float(value),
            "attack": self.attack_type.name.lower(),
            "weight": self.weight,
        }


class ThresholdDetector(BaseDetector):
    """固定阈值/统计检测器（§8.2 S1）。

    置信度语义：在本次命中的判据中，指向判定类别的权重占比（只命中同一类别时为 1.0）；
    未命中任何规则时判定为 NORMAL 且置信度为 1.0（即没有任何异常判据被触发）。
    """

    detector_id = "threshold"
    supported_context = ("Q", "N")
    version = "v1"

    @staticmethod
    def parse_rules(config: Mapping[str, Any]) -> list[ThresholdRule]:
        """从配置解析阈值规则表。

        Args:
            config: 含 ``rules`` 列表的配置映射，每项形如
                ``{"feature": "cn0_delta", "op": "<", "threshold": -6.0,
                "attack": "jamming", "weight": 1.0}``。

        Returns:
            规则列表。

        Raises:
            KeyError: 规则缺少必需字段。
            ValueError: 类别名或运算符非法。
        """
        rules: list[ThresholdRule] = []
        for raw in config.get("rules", ()) or ():
            attack = str(raw["attack"]).lower()
            if attack not in _ATTACK_ALIASES:
                raise ValueError(f"未知攻击类别 {attack!r}，允许：{sorted(_ATTACK_ALIASES)}")
            rules.append(
                ThresholdRule(
                    feature=str(raw["feature"]),
                    op=str(raw["op"]),
                    threshold=float(raw["threshold"]),
                    attack_type=_ATTACK_ALIASES[attack],
                    weight=float(raw.get("weight", 1.0)),
                )
            )
        return rules

    def run(
        self,
        data: Mapping[str, float],
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """按规则表评估当前窗口（§8.1）。

        Args:
            data: 特征名 → 数值的映射。
            context: 当前 GNSS Context（§7.2），S1 不依赖它，保留以统一接口。
            config: 含 ``rules`` 的配置；``None`` 表示无规则。

        Returns:
            DetectionResult：命中时指向对应类别，未命中时为 NORMAL。

        Note:
            上下文仅用于统一接口签名；本检测器不消费 Context。
        """
        rules = self.parse_rules(config or {})

        hits: list[tuple[ThresholdRule, float]] = []
        for rule in rules:
            value = data.get(rule.feature) if isinstance(data, Mapping) else None
            if value is None:
                continue  # 特征缺失不参与判别，交由数据质量层处理（§5.4）
            if rule.matches(float(value)):
                hits.append((rule, float(value)))

        if not hits:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=1.0,
                evidence={"matched_rules": [], "evaluated_rules": len(rules)},
                status=DetectorStatus.OK if rules else DetectorStatus.SKIPPED,
            )

        # 按类别累计权重，权重最大者作为判定结果（§12.4 冲突以证据形式暴露）
        weight_by_class: dict[AttackType, float] = {}
        for rule, _ in hits:
            weight_by_class[rule.attack_type] = weight_by_class.get(rule.attack_type, 0.0) + rule.weight

        chosen = max(weight_by_class, key=lambda cls: weight_by_class[cls])
        hit_weight_total = sum(weight_by_class.values())
        # 置信度 = 命中的判据中指向该类别的权重占比；只命中同一类别时为 1.0
        confidence = weight_by_class[chosen] / hit_weight_total if hit_weight_total > 0 else 1.0
        conflicts: Iterable[str] = sorted(
            cls.name.lower() for cls in weight_by_class if cls is not chosen
        )

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=chosen,
            confidence=min(1.0, confidence),
            evidence={
                "matched_rules": [rule.describe(value) for rule, value in hits],
                "conflicting_classes": list(conflicts),
                "evaluated_rules": len(rules),
            },
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )
