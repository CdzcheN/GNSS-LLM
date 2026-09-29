"""测试 src/fusion：§12.2 置信度加权融合与 §12.4 冲突判定。

对应开发文档
    §12.1 结果标准化、§12.2 基础融合、§12.4 冲突处理、§18.4 Multi-Strategy Fusion 消融。

覆盖要点
    - 单检测器输入时融合结果等于其自身判定；
    - §12.4 文档示例场景（C/N0 判 Jamming 0.82 vs PVT 判 Normal 0.71）必须判为冲突；
    - 空输入报错；非 OK 状态（ERROR/SKIPPED）不参与加权但仍留痕。
"""

from __future__ import annotations

import unittest

from src.detectors.base import AttackType, DetectionResult, DetectorStatus
from src.fusion.result_fusion import (
    FUSION_ORDER,
    StandardizedResult,
    detect_conflict,
    fuse,
)


class FusionOrderTest(unittest.TestCase):
    """融合步骤顺序须与 §12.3 一致。"""

    def test_fusion_order_follows_doc(self) -> None:
        self.assertEqual(
            FUSION_ORDER,
            ("rule_coordination", "confidence_weighting", "temporal_stability", "event_level"),
        )


class StandardizedResultTest(unittest.TestCase):
    """§12.1 的结果标准化映射。"""

    def test_from_result_keeps_doc_fields(self) -> None:
        result = DetectionResult("cno", AttackType.JAMMING, 0.8, evidence={"a": 1}, data_quality=0.9)
        standardized = StandardizedResult.from_result(result)
        self.assertEqual(standardized.detector_id, "cno")
        self.assertEqual(standardized.confidence, 0.8)
        self.assertEqual(standardized.data_quality, 0.9)
        self.assertIs(standardized.status, DetectorStatus.OK)


class FuseTest(unittest.TestCase):
    """§12.2 加权融合。"""

    def test_single_detector_keeps_its_verdict(self) -> None:
        fused = fuse([DetectionResult("cno", AttackType.JAMMING, 0.82)])
        self.assertEqual(fused.attack_type, AttackType.JAMMING)
        self.assertAlmostEqual(fused.confidence, 1.0)
        self.assertFalse(fused.conflict)

    def test_weights_shift_the_verdict(self) -> None:
        """提高 PVT 权重后，融合结论应随之改变（权重语义可验证）。"""
        pairs = [
            DetectionResult("cno", AttackType.JAMMING, 0.82),
            DetectionResult("pvt", AttackType.NORMAL, 0.71),
        ]
        fused = fuse(pairs, weights={"cno": 1.0, "pvt": 2.0})
        self.assertEqual(fused.attack_type, AttackType.NORMAL)

    def test_negative_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse([DetectionResult("cno", AttackType.JAMMING, 0.8)], weights={"cno": -1.0})

    def test_empty_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse([])

    def test_non_ok_results_do_not_score(self) -> None:
        fused = fuse([DetectionResult("x", AttackType.JAMMING, 0.9, status=DetectorStatus.ERROR)])
        self.assertEqual(fused.confidence, 0.0)
        self.assertEqual(fused.used_detectors, ("x",))

    def test_contributions_are_exposed(self) -> None:
        fused = fuse([DetectionResult("cno", AttackType.JAMMING, 0.82)])
        self.assertIn("cno", fused.contributions)


class ConflictTest(unittest.TestCase):
    """§12.4 冲突检测。"""

    DOC_SCENARIO = [
        DetectionResult("cno", AttackType.JAMMING, 0.82),
        DetectionResult("pvt", AttackType.NORMAL, 0.71),
    ]

    def test_doc_scenario_is_conflict(self) -> None:
        fused = fuse(self.DOC_SCENARIO)
        self.assertTrue(fused.conflict)
        self.assertIn("pvt", fused.conflicting_detectors)
        self.assertTrue(detect_conflict(self.DOC_SCENARIO))

    def test_clear_winner_is_not_conflict(self) -> None:
        clear = [
            DetectionResult("cno", AttackType.JAMMING, 0.95),
            DetectionResult("pvt", AttackType.NORMAL, 0.30),
        ]
        self.assertFalse(detect_conflict(clear))

    def test_single_class_is_not_conflict(self) -> None:
        agreement = [
            DetectionResult("cno", AttackType.JAMMING, 0.8),
            DetectionResult("spectrum", AttackType.JAMMING, 0.6),
        ]
        self.assertFalse(detect_conflict(agreement))

    def test_low_confidence_disagreement_ignored(self) -> None:
        low = [
            DetectionResult("cno", AttackType.JAMMING, 0.9),
            DetectionResult("pvt", AttackType.NORMAL, 0.2),
        ]
        self.assertFalse(detect_conflict(low))


if __name__ == "__main__":
    unittest.main()
