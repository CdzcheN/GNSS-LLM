"""测试 src/eval：五层指标键、混淆矩阵 / P-R-F1 / Macro-F1、事件级匹配与消融定义。

对应开发文档
    §17.1–§17.5 五层指标、§17.2（Jamming 必须单列）、§17.3 事件级指标、
    §18.1–§18.8 消融定义。

覆盖要点
    - 五层指标键齐备且与 §17 一致，Jamming 有独立键；
    - 混淆矩阵与各类 P/R/F1 的数值正确（用可手算的固定样例）；
    - 事件匹配采用时间重叠规则，邻接区间不算重叠；
    - 八个消融项与 §18 小节一一对应，且对照组校验能拦住非法取值。
"""

from __future__ import annotations

import unittest

from src.eval.ablation import (
    ABLATION_KEYS,
    get_ablation,
    summary,
    validate_arms,
)
from src.eval.metrics import (
    CLASS_LEVEL_METRICS,
    EVENT_LEVEL_METRICS,
    SECOND_LEVEL_METRICS,
    STRATEGY_LEVEL_METRICS,
    SYSTEM_LEVEL_METRICS,
    Interval,
    accuracy,
    classification_report,
    confusion_matrix,
    event_level_metrics,
    macro_f1,
    match_events,
    per_class_metrics,
)

#: 可手算的固定样例：三态各 2 个样本，其中 1 个 Jamming 被判为 Spoofing。
Y_TRUE = (0, 0, 1, 1, 2, 2)
Y_PRED = (0, 0, 1, 1, 1, 2)


class MetricKeysTest(unittest.TestCase):
    """§17 五层指标键。"""

    def test_all_five_layers_defined(self) -> None:
        for keys in (
            SECOND_LEVEL_METRICS,
            CLASS_LEVEL_METRICS,
            EVENT_LEVEL_METRICS,
            STRATEGY_LEVEL_METRICS,
            SYSTEM_LEVEL_METRICS,
        ):
            self.assertTrue(keys, "某一层指标键为空（§17）")

    def test_jamming_reported_separately(self) -> None:
        """§17.2 特别要求 Jamming 单独报告。"""
        self.assertIn("jamming_p_r_f1", CLASS_LEVEL_METRICS)
        self.assertIn("spoofing_p_r_f1", CLASS_LEVEL_METRICS)

    def test_system_level_contains_int8_delta(self) -> None:
        self.assertIn("int8_delta", SYSTEM_LEVEL_METRICS)


class ConfusionMatrixTest(unittest.TestCase):
    """§17.1 混淆矩阵基础。"""

    def test_counts(self) -> None:
        matrix = confusion_matrix(Y_TRUE, Y_PRED)
        self.assertEqual(matrix[0][0], 2)
        self.assertEqual(matrix[2][1], 1)
        self.assertEqual(matrix[2][2], 1)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0])

    def test_unknown_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confusion_matrix([9], [0])


class ClassificationMetricTest(unittest.TestCase):
    """§17.1 / §17.2 逐秒与类别级指标。"""

    def test_accuracy(self) -> None:
        matrix = confusion_matrix(Y_TRUE, Y_PRED)
        self.assertAlmostEqual(accuracy(matrix), 5 / 6, places=4)

    def test_per_class_metrics(self) -> None:
        matrix = confusion_matrix(Y_TRUE, Y_PRED)
        spoofing = per_class_metrics(matrix, 1)
        self.assertAlmostEqual(spoofing["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(spoofing["recall"], 1.0, places=4)
        jamming = per_class_metrics(matrix, 2)
        self.assertAlmostEqual(jamming["precision"], 1.0, places=4)
        self.assertAlmostEqual(jamming["recall"], 0.5, places=4)

    def test_macro_f1_averages_classes(self) -> None:
        matrix = confusion_matrix(Y_TRUE, Y_PRED)
        # normal 1.0、spoofing 0.8、jamming 0.6667 → 均值 0.8222
        self.assertAlmostEqual(macro_f1(matrix), 0.8222, places=4)

    def test_report_exposes_class_level_keys(self) -> None:
        report = classification_report(Y_TRUE, Y_PRED)
        self.assertIn("jamming", report["per_class"])
        self.assertIn("jamming_p_r_f1", report)
        self.assertAlmostEqual(report["accuracy"], 5 / 6, places=4)


class EventMetricTest(unittest.TestCase):
    """§17.3 事件级指标。"""

    def test_overlap_rule(self) -> None:
        self.assertTrue(Interval(100, 200).overlaps(Interval(150, 220)))
        self.assertFalse(Interval(0, 10).overlaps(Interval(10, 20)))

    def test_match_reports_missed_and_false_alarms(self) -> None:
        truth = [Interval(100, 200, "jamming"), Interval(300, 400, "spoofing")]
        predicted = [Interval(105, 195, "jamming"), Interval(500, 600, "spoofing")]
        result = match_events(predicted, truth)
        self.assertEqual(len(result["matched"]), 1)
        self.assertEqual(len(result["missed"]), 1)
        self.assertEqual(len(result["false_alarms"]), 1)

    def test_event_level_metrics_values(self) -> None:
        truth = [Interval(100, 200), Interval(300, 400)]
        predicted = [Interval(110, 190)]
        metrics = event_level_metrics(predicted, truth)
        self.assertAlmostEqual(metrics["event_recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["event_detection_rate"], 1.0, places=4)
        self.assertEqual(metrics["missed_events"], 1)
        self.assertEqual(metrics["false_alarm_events"], 0)
        self.assertAlmostEqual(metrics["detection_delay"], 10.0, places=4)

    def test_empty_inputs_do_not_divide_by_zero(self) -> None:
        metrics = event_level_metrics([], [])
        self.assertEqual(metrics["event_recall"], 0.0)
        self.assertEqual(metrics["event_level_precision"], 0.0)


class AblationTest(unittest.TestCase):
    """§18 消融定义。"""

    def test_eight_ablations_matching_doc_sections(self) -> None:
        self.assertEqual(len(ABLATION_KEYS), 8)
        self.assertEqual(get_ablation("context").section, "§18.1")
        self.assertEqual(get_ablation("dynamic_selection").section, "§18.2")
        self.assertEqual(get_ablation("llm").section, "§18.8")

    def test_arms_follow_doc(self) -> None:
        self.assertEqual(get_ablation("context").arms, ("no_context", "with_context"))
        self.assertEqual(get_ablation("modality").arms, ("A", "A+B", "A+C", "A+B+C"))
        self.assertEqual(get_ablation("window").arms, ("10", "30", "60", "120"))
        self.assertEqual(
            get_ablation("agent").arms,
            ("rule_router", "learning_router", "agent_router"),
        )

    def test_unknown_ablation_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_ablation("nope")

    def test_validate_arms(self) -> None:
        self.assertEqual(validate_arms("context", ["with_context"]), ("with_context",))
        with self.assertRaises(ValueError):
            validate_arms("context", ["unknown_arm"])

    def test_empty_arms_returns_doc_default(self) -> None:
        self.assertEqual(validate_arms("context", []), get_ablation("context").arms)

    def test_summary_is_serializable(self) -> None:
        entries = summary()
        self.assertEqual(len(entries), 8)
        self.assertIn("switch", entries[0])


if __name__ == "__main__":
    unittest.main()
