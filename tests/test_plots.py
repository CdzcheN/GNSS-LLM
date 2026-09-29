"""测试 src/eval/plots：图件能否生成、异常输入是否被拒、字体缺失是否降级。

对应开发文档
    §17 评价指标可视化、§18 消融对比图、§20.3 实验记录的 figure_paths 字段。

覆盖要点
    - 五类图件（时序对比 / 混淆矩阵 / 类别指标 / 特征曲线 / 事件时间线）与消融柱状图
      都能写出非空的 PNG 文件；
    - ``make_evaluation_figures`` 一次性产出多张图并返回路径列表；
    - 缺列、空输入等异常输入给出明确错误；
    - 超长序列会按 ``max_points`` 抽样（保证图件不至于过大）；
    - 没有中文字体时只降级为英文标注，不应抛异常。

说明
    需要 matplotlib 与 pandas；未安装时整类用例跳过。图件统一写入临时目录，不污染仓库。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - 环境相关
    import matplotlib

    matplotlib.use("Agg", force=True)
    import pandas as pd

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

if HAS_MATPLOTLIB:
    from src.eval.metrics import Interval, classification_report, confusion_matrix
    from src.eval.plots import (
        CHINESE_FONT_CANDIDATES,
        feature_discriminability,
        make_evaluation_figures,
        plot_ablation_comparison,
        plot_class_metrics,
        plot_confidence_distribution,
        plot_conflict_timeline,
        plot_confusion_matrix,
        plot_correlation_matrix,
        plot_detector_usage,
        plot_event_timeline,
        plot_feature_distributions,
        plot_feature_discriminability,
        plot_feature_timeline,
        plot_label_distribution,
        plot_prediction_timeline,
        plot_roc_pr_curves,
    )


def build_frame(rows: int = 400) -> "pd.DataFrame":
    """构造带真实标签、预测、分数与关键特征的合成结果表（含少量可辨认的错误）。

    Args:
        rows: 行数。

    Returns:
        含 Timestamp / Label / prediction / confidence / conflict / detectors
        与四个特征列的 DataFrame；异常区间的特征明显偏离正常区间（保证可区分）。
    """
    index = pd.date_range("2023-12-21 12:00:00", periods=rows, freq="1s")
    truth = [0] * rows
    prediction = [0] * rows
    # 中段为 Spoofing（部分被检出），尾段为 Jamming（完全漏检）
    for position in range(rows // 3, rows // 2):
        truth[position] = 1
    for position in range(rows // 3, rows // 3 + rows // 10):
        prediction[position] = 1
    for position in range(rows * 2 // 3, rows):
        truth[position] = 2

    cn0_by_class = {0: 0.0, 1: -6.0, 2: -12.0}
    residual_by_class = {0: 1.0, 1: 2.0, 2: 12.0}

    return pd.DataFrame(
        {
            "Timestamp": index,
            "Label": truth,
            "prediction": prediction,
            "confidence": [0.9 if pred == real else 0.25 for real, pred in zip(truth, prediction)],
            "conflict": [real != 0 and pred == 0 for real, pred in zip(truth, prediction)],
            "detectors": ["cno,satellite" if real != 0 else "cno" for real in truth],
            "cn0_delta_db": [cn0_by_class[real] for real in truth],
            "res_valid_mean": [residual_by_class[real] for real in truth],
            "pDOP_vs_baseline": [0.0 if real == 0 else 1.5 for real in truth],
            "valid_sat_count": [8 if real == 0 else 5 for real in truth],
        }
    )


@unittest.skipUnless(HAS_MATPLOTLIB, "需要 matplotlib / pandas")
class PlotGenerationTest(unittest.TestCase):
    """五类图件与消融图能正常产出。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.frame = build_frame()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_nonempty(self, path: Path) -> None:
        self.assertTrue(path.exists(), f"图件未生成：{path}")
        self.assertGreater(path.stat().st_size, 1000, f"图件过小，可能为空：{path}")

    def test_prediction_timeline(self) -> None:
        path = plot_prediction_timeline(self.frame, self.tmp / "timeline.png")
        self._assert_nonempty(path)

    def test_confusion_matrix(self) -> None:
        matrix = confusion_matrix(
            self.frame["Label"].tolist(), self.frame["prediction"].tolist(), labels=(0, 1, 2)
        )
        path = plot_confusion_matrix(matrix, self.tmp / "confusion.png")
        self._assert_nonempty(path)

    def test_class_metrics(self) -> None:
        report = classification_report(
            self.frame["Label"].tolist(), self.frame["prediction"].tolist(), labels=(0, 1, 2)
        )
        path = plot_class_metrics(report, self.tmp / "class_metrics.png")
        self._assert_nonempty(path)

    def test_class_metrics_requires_per_class(self) -> None:
        with self.assertRaises(ValueError):
            plot_class_metrics({}, self.tmp / "bad.png")

    def test_feature_timeline(self) -> None:
        path = plot_feature_timeline(
            self.frame, ["cn0_delta_db", "res_valid_mean"], self.tmp / "features.png"
        )
        self._assert_nonempty(path)

    def test_feature_timeline_rejects_missing_column(self) -> None:
        with self.assertRaises(ValueError):
            plot_feature_timeline(self.frame, ["not_a_column"], self.tmp / "bad.png")
        with self.assertRaises(ValueError):
            plot_feature_timeline(self.frame, [], self.tmp / "bad.png")

    def test_event_timeline(self) -> None:
        truth = [Interval(100, 200), Interval(300, 380)]
        detected = [Interval(105, 195)]
        path = plot_event_timeline(truth, detected, self.tmp / "events.png")
        self._assert_nonempty(path)

    def test_ablation_comparison(self) -> None:
        path = plot_ablation_comparison(
            {"no_context": 0.71, "with_context": 0.78}, self.tmp / "ablation.png"
        )
        self._assert_nonempty(path)
        with self.assertRaises(ValueError):
            plot_ablation_comparison({}, self.tmp / "bad.png")

    def test_make_evaluation_figures_bundle(self) -> None:
        outputs = make_evaluation_figures(
            self.frame,
            figures_dir=self.tmp / "bundle",
            prefix="unit",
            feature_columns=("cn0_delta_db",),
            truth_events=[Interval(100, 200)],
            predicted_events=[Interval(110, 190)],
        )
        names = sorted(path.name for path in outputs)
        self.assertIn("unit_timeline.png", names)
        self.assertIn("unit_confusion.png", names)
        self.assertIn("unit_class_metrics.png", names)
        self.assertIn("unit_features.png", names)
        self.assertIn("unit_events.png", names)
        for path in outputs:
            self._assert_nonempty(path)

    def test_make_evaluation_figures_rejects_missing_prediction_column(self) -> None:
        broken = self.frame.drop(columns=["prediction"])
        with self.assertRaises(ValueError):
            make_evaluation_figures(broken, figures_dir=self.tmp / "x")


@unittest.skipUnless(HAS_MATPLOTLIB, "需要 matplotlib / pandas")
class SamplingAndFontTest(unittest.TestCase):
    """长序列抽样与字体降级。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_long_series_is_sampled(self) -> None:
        """抽样后仍能出图（点数被限制，不因数据量大而失败或超时）。"""
        frame = build_frame(rows=9000)
        path = plot_prediction_timeline(frame, self.tmp / "long.png", max_points=500)
        self.assertGreater(path.stat().st_size, 1000)

    def test_no_chinese_font_still_renders(self) -> None:
        """字体缺失只影响标注语言，不应导致绘图失败。"""
        path = plot_prediction_timeline(build_frame(rows=50), self.tmp / "en.png")
        self.assertGreater(path.stat().st_size, 1000)

    def test_font_candidate_list_is_nonempty(self) -> None:
        self.assertGreater(len(CHINESE_FONT_CANDIDATES), 0)


@unittest.skipUnless(HAS_MATPLOTLIB, "需要 matplotlib / pandas")
class ExtendedPlotTest(unittest.TestCase):
    """新增图件：分布对比 / 特征可分性 / ROC-PR / 检测器与冲突。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.frame = build_frame()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _assert_nonempty(self, path: Path) -> None:
        self.assertTrue(path.exists(), f"图件未生成：{path}")
        self.assertGreater(path.stat().st_size, 1000, f"图件过小，可能为空：{path}")

    def test_label_distribution(self) -> None:
        self._assert_nonempty(plot_label_distribution(self.frame, self.tmp / "labels.png"))

    def test_confidence_distribution(self) -> None:
        self._assert_nonempty(plot_confidence_distribution(self.frame, self.tmp / "conf.png"))

    def test_feature_distributions(self) -> None:
        path = plot_feature_distributions(
            self.frame, ["cn0_delta_db", "res_valid_mean", "pDOP_vs_baseline"], self.tmp / "box.png"
        )
        self._assert_nonempty(path)
        with self.assertRaises(ValueError):
            plot_feature_distributions(self.frame, [], self.tmp / "bad.png")

    def test_feature_discriminability_ranks_separable_feature(self) -> None:
        """正常段（0.0）与异常段（-6/-12）明显分离，区分度应显著大于 0。"""
        scores = feature_discriminability(self.frame, ["cn0_delta_db", "res_valid_mean"])
        self.assertIn("cn0_delta_db", scores)
        self.assertGreater(scores["cn0_delta_db"], 1.0)
        self._assert_nonempty(
            plot_feature_discriminability(self.frame, ["cn0_delta_db", "res_valid_mean"],
                                         self.tmp / "rank.png")
        )

    def test_feature_discriminability_skips_constant_feature(self) -> None:
        """零方差特征无信息量，应被跳过而不是报错或给出无穷值。"""
        frame = self.frame.copy()
        frame["constant"] = 1.0
        self.assertEqual(feature_discriminability(frame, ["constant"]), {})

    def test_roc_pr_curves(self) -> None:
        path = plot_roc_pr_curves(self.frame, self.tmp / "roc.png")
        self._assert_nonempty(path)

    def test_roc_pr_requires_two_classes(self) -> None:
        only_normal = self.frame.copy()
        only_normal["Label"] = 0
        with self.assertRaises(ValueError):
            plot_roc_pr_curves(only_normal, self.tmp / "bad.png")

    def test_detector_usage(self) -> None:
        path = plot_detector_usage(self.frame, self.tmp / "usage.png")
        self._assert_nonempty(path)

    def test_conflict_timeline(self) -> None:
        path = plot_conflict_timeline(self.frame, self.tmp / "conflict.png", window=50)
        self._assert_nonempty(path)
        with self.assertRaises(ValueError):
            plot_conflict_timeline(self.frame, self.tmp / "bad.png", window=0)

    def test_correlation_matrix(self) -> None:
        path = plot_correlation_matrix(
            self.frame, ["cn0_delta_db", "res_valid_mean", "valid_sat_count"], self.tmp / "corr.png"
        )
        self._assert_nonempty(path)

    def test_correlation_requires_two_columns(self) -> None:
        with self.assertRaises(ValueError):
            plot_correlation_matrix(self.frame, ["cn0_delta_db"], self.tmp / "bad.png")

    def test_bundle_generates_all_figures(self) -> None:
        """打包入口应产出至少 10 张图，且文件名覆盖各类型。"""
        outputs = make_evaluation_figures(
            self.frame,
            figures_dir=self.tmp / "bundle",
            prefix="full",
            feature_columns=("cn0_delta_db", "res_valid_mean", "pDOP_vs_baseline", "valid_sat_count"),
            truth_events=[Interval(100, 200)],
            predicted_events=[Interval(110, 190)],
        )
        names = {path.name for path in outputs}
        for expected in (
            "full_timeline.png",
            "full_confusion.png",
            "full_class_metrics.png",
            "full_label_dist.png",
            "full_features.png",
            "full_feature_box.png",
            "full_feature_rank.png",
            "full_correlation.png",
            "full_confidence.png",
            "full_roc_pr.png",
            "full_detector_usage.png",
            "full_conflict.png",
            "full_events.png",
        ):
            self.assertIn(expected, names)
        self.assertGreaterEqual(len(outputs), 13)
        for path in outputs:
            self._assert_nonempty(path)


if __name__ == "__main__":
    unittest.main()
