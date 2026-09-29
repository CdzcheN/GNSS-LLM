"""测试 src/detectors：三态契约、DetectionResult 与 S1 阈值规则引擎。

对应开发文档
    §1.3 三态识别、§8.1 统一接口、§8.2 S1 固定阈值/统计检测、§2.2 证据可追溯。

覆盖要点
    - 三态取值为 0/1/2；
    - DetectionResult 置信度越界必须拒绝；
    - 阈值规则：命中取权重最大类别、未命中判 NORMAL、特征缺失跳过、无规则标记 SKIPPED。
"""

from __future__ import annotations

import unittest

from src.detectors.base import AttackType, DetectionResult, DetectorStatus
from src.detectors.threshold import ThresholdDetector, ThresholdRule


class AttackTypeTest(unittest.TestCase):
    """三态标签必须与开发文档 §1.3 一致。"""

    def test_three_state_values(self) -> None:
        self.assertEqual(
            (int(AttackType.NORMAL), int(AttackType.SPOOFING), int(AttackType.JAMMING)),
            (0, 1, 2),
        )


class DetectionResultTest(unittest.TestCase):
    """DetectionResult 的结构与取值校验（§8.1）。"""

    def test_confidence_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DetectionResult("t", AttackType.JAMMING, 1.5)

    def test_to_dict_contains_all_doc_fields(self) -> None:
        payload = DetectionResult("t", AttackType.SPOOFING, 0.9).to_dict()
        for key in (
            "detector_id", "attack_type", "confidence", "evidence",
            "timestamp", "latency_ms", "data_quality", "status",
        ):
            self.assertIn(key, payload)


class ThresholdRuleTest(unittest.TestCase):
    """阈值规则自身的合法性校验。"""

    def test_invalid_operator_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdRule("f", "==", 1.0, AttackType.JAMMING)

    def test_negative_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdRule("f", "<", 1.0, AttackType.JAMMING, weight=-1.0)

    def test_matches(self) -> None:
        rule = ThresholdRule("cn0", "<", -6.0, AttackType.JAMMING)
        self.assertTrue(rule.matches(-7.0))
        self.assertFalse(rule.matches(-5.0))


class ThresholdDetectorTest(unittest.TestCase):
    """S1 检测器行为（判据全部来自配置）。"""

    RULES = {
        "rules": [
            {"feature": "cn0_delta_db", "op": "<", "threshold": -6, "attack": "jamming", "weight": 2.0},
            {"feature": "pdop_delta", "op": ">", "threshold": 3, "attack": "spoofing", "weight": 1.0},
        ]
    }

    def test_two_classes_hit_selects_highest_weight(self) -> None:
        result = ThresholdDetector().run({"cn0_delta_db": -8.0, "pdop_delta": 4.0}, None, self.RULES)
        self.assertEqual(result.attack_type, AttackType.JAMMING)
        self.assertAlmostEqual(result.confidence, 2.0 / 3.0)
        self.assertIn("spoofing", result.evidence["conflicting_classes"])

    def test_single_hit(self) -> None:
        result = ThresholdDetector().run({"cn0_delta_db": -8.0}, None, self.RULES)
        self.assertEqual(result.attack_type, AttackType.JAMMING)
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_no_hit_returns_normal(self) -> None:
        result = ThresholdDetector().run({"cn0_delta_db": 0.0}, None, self.RULES)
        self.assertEqual(result.attack_type, AttackType.NORMAL)
        self.assertEqual(result.confidence, 1.0)
        self.assertIs(result.status, DetectorStatus.OK)

    def test_missing_feature_is_skipped_not_matched(self) -> None:
        result = ThresholdDetector().run({}, None, self.RULES)
        self.assertEqual(result.attack_type, AttackType.NORMAL)
        self.assertEqual(result.evidence["matched_rules"], [])

    def test_empty_rules_marked_skipped(self) -> None:
        result = ThresholdDetector().run({"any": 1.0}, None, {})
        self.assertIs(result.status, DetectorStatus.SKIPPED)

    def test_invalid_attack_alias_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ThresholdDetector().run({}, None, {"rules": [{"feature": "f", "op": "<", "threshold": 1, "attack": "unknown"}]})


if __name__ == "__main__":
    unittest.main()
