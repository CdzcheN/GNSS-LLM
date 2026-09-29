"""测试事件级切分与阈值标定工具（§16.2、§16.3、§8.2）。

对应开发文档
    §16.2 数据划分（正常数据时间段划分；1221 按攻击事件划分；Leave-One-Event-Out）、
    §16.3 防泄漏规范、§8.2 检测器判据、§8.3 策略适用性。

覆盖要点
    - 事件段识别：连续同标签被正确切段；
    - ``split_by_events``：划分互不重叠、覆盖全部行、**验证集含所有出现过的类别**
      （这是修复“验证集全 Normal 导致早停失效”的关键断言）；
    - 每类至少 1 段留给训练，单一事件段不做切分；
    - ``windows_from_segments``：按段建窗，不产生跨段的假窗口；
    - 标定工具：``scan_threshold`` 在构造数据上给出正确工作点、
      ``auc_score`` 对完全可分数据接近 1.0、对随机数据接近 0.5，
      且 **NaN 会被按行剔除**（变化率类特征首行为 NaN 的历史缺陷）。
"""

from __future__ import annotations

import unittest

try:  # pragma: no cover - 环境相关
    import numpy as np
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False

from src.data import schema
from src.train.dataset import (
    EventSplit,
    find_event_segments,
    split_by_events,
    windows_from_segments,
)

if HAS_PANDAS:
    from scripts.calibrate_thresholds import auc_score, scan_threshold


def build_labeled_frame(segments: "list[tuple[int, int]]") -> "pd.DataFrame":
    """按给定区间构造带标签的最小特征表。

    Args:
        segments: ``(长度, 标签)`` 序列，按时间顺序拼接。

    Returns:
        含 Timestamp / Label / 一个特征列的 DataFrame。
    """
    labels: list[int] = []
    for length, label in segments:
        labels.extend([label] * length)
    rows = len(labels)
    index = pd.date_range("2023-12-21 12:00:00", periods=rows, freq="1s")
    return pd.DataFrame(
        {
            schema.TIME_COLUMN: index,
            schema.DAY_COLUMN: schema.THREE_STATE_DAY,
            schema.HOUR_COLUMN: 12,
            schema.LABEL_COLUMN: labels,
            "feature": [float(label) for label in labels],
        }
    )


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class EventSegmentTest(unittest.TestCase):
    """事件段识别。"""

    def test_segments_are_split_by_label_change(self) -> None:
        frame = build_labeled_frame([(10, 0), (5, 1), (7, 0), (3, 2)])
        segments = find_event_segments(frame, label_column=schema.LABEL_COLUMN)
        self.assertEqual(len(segments[0]), 2)   # 两段 Normal
        self.assertEqual(len(segments[1]), 1)
        self.assertEqual(len(segments[2]), 1)
        self.assertEqual(segments[1][0], (10, 14))

    def test_missing_label_column_rejected(self) -> None:
        frame = build_labeled_frame([(5, 0)]).drop(columns=[schema.LABEL_COLUMN])
        with self.assertRaises(ValueError):
            find_event_segments(frame, label_column=schema.LABEL_COLUMN)


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class EventSplitTest(unittest.TestCase):
    """§16.2 事件级划分。"""

    def setUp(self) -> None:
        # 模拟真实结构：Normal 全天散布，Spoofing 与 Jamming 各有多个事件段
        self.frame = build_labeled_frame([
            (100, 0), (20, 1), (30, 0), (25, 1), (40, 0), (18, 1),
            (50, 0), (12, 2), (35, 0), (11, 2), (45, 0), (13, 2), (60, 0),
        ])

    def test_validation_contains_every_class(self) -> None:
        """核心断言：修复“按时间切分导致验证集全 Normal”的缺陷。"""
        split = split_by_events(self.frame, label_column=schema.LABEL_COLUMN)
        validation = self.frame.iloc[split.indices("validation")]
        present = set(validation[schema.LABEL_COLUMN].unique())
        self.assertEqual(present, {0, 1, 2}, f"验证集类别不全：{present}")

    def test_train_contains_every_class(self) -> None:
        split = split_by_events(self.frame, label_column=schema.LABEL_COLUMN)
        train = self.frame.iloc[split.indices("train")]
        self.assertEqual(set(train[schema.LABEL_COLUMN].unique()), {0, 1, 2})

    def test_parts_are_disjoint_and_complete(self) -> None:
        split = split_by_events(self.frame, label_column=schema.LABEL_COLUMN)
        train_index = set(split.indices("train").tolist())
        validation_index = set(split.indices("validation").tolist())
        self.assertEqual(train_index & validation_index, set())
        self.assertEqual(len(train_index | validation_index), len(self.frame))

    def test_counts_match_segments(self) -> None:
        split = split_by_events(self.frame, label_column=schema.LABEL_COLUMN)
        self.assertEqual(split.train_rows + split.validation_rows, len(self.frame))
        self.assertGreater(split.train_rows, 0)
        self.assertGreater(split.validation_rows, 0)

    def test_single_event_class_is_not_split(self) -> None:
        frame = build_labeled_frame([(50, 0), (20, 1), (30, 0)])
        split = split_by_events(frame, label_column=schema.LABEL_COLUMN)
        # 只有一个 Spoofing 段 → 整体归训练，避免训练侧缺失该类别
        self.assertEqual(split.details["Spoofing"]["validation"], 0)
        self.assertEqual(split.details["Spoofing"]["train"], 1)

    def test_summary_is_serializable(self) -> None:
        split = split_by_events(self.frame, label_column=schema.LABEL_COLUMN)
        summary = split.summary()
        for key in ("train_rows", "validation_rows", "train_segments", "per_class"):
            self.assertIn(key, summary)

    def test_indices_reject_unknown_part(self) -> None:
        split = EventSplit(train_segments=((0, 9),), validation_segments=((10, 19),))
        with self.assertRaises(ValueError):
            split.indices("test")


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class WindowsFromSegmentsTest(unittest.TestCase):
    """按段建窗，避免跨段假窗口。"""

    def test_windows_do_not_cross_segments(self) -> None:
        frame = build_labeled_frame([(30, 0), (30, 2)])
        # 两段都被切成 10 长度窗口；若不分段，段边界处会出现“跨段窗口”
        X, y = windows_from_segments(
            frame, ((0, 29), (30, 59)), ["feature"], window_s=10, stride_s=10
        )
        self.assertEqual(X.shape[0], 6)          # 每段 3 个窗口
        self.assertEqual(set(np.unique(y).tolist()), {0.0, 2.0})
        # 跨段窗口会混合标签；这里第 3 个窗口（末段窗口）标签应全是 0
        self.assertEqual(X[0:3][:, :, 0].max(), 0.0)
        self.assertEqual(X[3:6][:, :, 0].min(), 2.0)

    def test_short_segments_are_skipped(self) -> None:
        frame = build_labeled_frame([(5, 0), (20, 2)])
        X, y = windows_from_segments(
            frame, ((0, 4), (5, 24)), ["feature"], window_s=10, stride_s=10
        )
        # 第一段只有 5 行（< window_s=10）被整段跳过；第二段 20 行按步长 10 得到 2 个窗口
        self.assertEqual(X.shape[0], 2)

    def test_all_segments_too_short_reports_clearly(self) -> None:
        frame = build_labeled_frame([(3, 0), (4, 2)])
        with self.assertRaises(ValueError) as caught:
            windows_from_segments(frame, ((0, 2), (3, 6)), ["feature"], window_s=10)
        self.assertIn("window_s", str(caught.exception))


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class ThresholdScanTest(unittest.TestCase):
    """阈值标定的数值正确性。"""

    def test_auc_near_one_for_separable_feature(self) -> None:
        values = np.r_[np.zeros(50), np.full(50, 10.0)]
        positive = np.r_[np.zeros(50, dtype=bool), np.ones(50, dtype=bool)]
        self.assertAlmostEqual(auc_score(values, positive, "ge"), 1.0, places=6)
        self.assertAlmostEqual(auc_score(values, positive, "le"), 0.0, places=6)

    def test_auc_near_half_for_uninformative_feature(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=2000)
        positive = rng.random(2000) > 0.5
        self.assertAlmostEqual(auc_score(values, positive, "ge"), 0.5, delta=0.08)

    def test_auc_nan_when_single_class(self) -> None:
        values = np.arange(10, dtype="float64")
        positive = np.zeros(10, dtype=bool)
        self.assertTrue(np.isnan(auc_score(values, positive, "ge")))

    def test_scan_finds_separating_threshold(self) -> None:
        values = np.r_[np.zeros(100), np.full(100, 50.0)]
        positive = np.r_[np.zeros(100, dtype=bool), np.ones(100, dtype=bool)]
        best = scan_threshold(values, positive, "ge", quantiles=50)
        self.assertGreater(best["f1"], 0.99)
        self.assertGreater(best["threshold"], 0.0)
        self.assertLessEqual(best["threshold"], 50.0)

    def test_scan_direction_le_handles_lower_is_worse(self) -> None:
        values = np.r_[np.full(100, -9.0), np.zeros(100)]
        positive = np.r_[np.ones(100, dtype=bool), np.zeros(100, dtype=bool)]
        best = scan_threshold(values, positive, "le", quantiles=50)
        self.assertGreater(best["f1"], 0.99)

    def test_scan_rejects_bad_direction(self) -> None:
        with self.assertRaises(ValueError):
            scan_threshold(np.arange(10, dtype="float64"), np.ones(10, dtype=bool), "up")


if __name__ == "__main__":
    unittest.main()
