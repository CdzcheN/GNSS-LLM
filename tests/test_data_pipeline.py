"""测试数据管道：schema / parser / alignment / quality / features / context / detectors。

对应开发文档
    §5.2 数据资产、§5.3 模态定义、§5.4 数据质量、§6.1–§6.4 处理与特征工程、
    §7 Context、§8.2 检测器判据、§8.3 策略适用性、§16.2 数据划分。

覆盖要点
    - schema 的 112 列与 108 个数值特征，顺序与实测表头一致；
    - 解析、时间排序、重复历元清理与缺口统计；
    - 卫星掩码（``CNO > 0.5``）与数据质量评分；
    - 三类特征列的正确性与**因果性**（变化率不使用未来行，§16.3）；
    - Context 六分量与 §8.3 的 flags 派生；
    - 规则检测器的命中 / 不命中分支；
    - 窗口张量形状与“标签取窗口末行”的因果约定。

说明
    本测试使用**合成数据**（按 schema 生成），不依赖 ``data/`` 下的真实数据集，
    因此可在无数据环境下秒级运行；真实数据的端到端检查见
    ``scripts/run_pipeline_check.py``。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - 环境相关
    import numpy as np
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False

from src.data import alignment, parser, quality, schema

if HAS_PANDAS:
    from src.context.context_encoder import ContextEncoder, FlagThresholds, derive_flags
    from src.detectors.cno import CnoDetector
    from src.detectors.observation import ObservationDetector
    from src.detectors.pvt import PvtDetector
    from src.detectors.satellite import SatelliteDetector
    from src.features import navigation, observation, signal
    from src.train.dataset import build_windows


def make_frame(
    rows: int = 300,
    start: str = "2023-09-12 00:00:00",
    cno: float = 30.0,
    residual: float = 1.0,
    label: int = 0,
) -> "pd.DataFrame":
    """按 schema 生成合成特征表（可直接喂给各模块）。

    Args:
        rows: 行数。
        start: 起始时间（1 Hz 递增）。
        cno: 所有卫星的 C/N0 初值。
        residual: 所有卫星的伪距残差初值。
        label: 标签列取值。

    Returns:
        含 112 列的 DataFrame。
    """
    index = pd.date_range(start, periods=rows, freq="1s")
    data: dict[str, object] = {
        schema.TIME_COLUMN: index,
        schema.DAY_COLUMN: 12,
        schema.HOUR_COLUMN: 0,
    }
    for name in schema.PVT_COLUMNS:
        data[name] = 1.0
    for name in schema.CNO_COLUMNS:
        data[name] = cno
    for name in schema.RES_COLUMNS:
        data[name] = residual
    for name in schema.ELEV_COLUMNS:
        data[name] = 45.0
    data[schema.LABEL_COLUMN] = label
    return pd.DataFrame(data)[list(schema.EXPECTED_COLUMNS)]


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class SchemaTest(unittest.TestCase):
    """§5.3 的列契约。"""

    def test_column_counts(self) -> None:
        self.assertEqual(schema.EXPECTED_COLUMN_COUNT, 112)
        self.assertEqual(schema.EXPECTED_FEATURE_COUNT, 108)
        self.assertEqual(len(schema.PRN_LIST), 32)

    def test_assert_columns_accepts_expected(self) -> None:
        schema.assert_columns(schema.EXPECTED_COLUMNS)

    def test_assert_columns_rejects_wrong_order(self) -> None:
        swapped = list(schema.EXPECTED_COLUMNS)
        swapped[3], swapped[4] = swapped[4], swapped[3]
        with self.assertRaises(ValueError):
            schema.assert_columns(swapped)

    def test_split_disjoint_and_covers_all_days(self) -> None:
        schema.assert_split_disjoint()
        days = sorted(day for days in schema.SPLIT_BY_DAY.values() for day in days)
        self.assertEqual(days, list(range(12, 31)))


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class ParserAndAlignmentTest(unittest.TestCase):
    """§6.1 前四步：解析 → 排序 → 去重 → 缺口统计。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_validates_and_parses_time(self) -> None:
        path = self.tmp / "sample.csv"
        make_frame(50).to_csv(path, index=False)
        frame = parser.load_features_csv(path)
        self.assertEqual(len(frame), 50)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame[schema.TIME_COLUMN]))
        self.assertIn(parser.SOURCE_COLUMN, frame.columns)

    def test_load_rejects_wrong_columns(self) -> None:
        path = self.tmp / "bad.csv"
        make_frame(10).drop(columns=[schema.LABEL_COLUMN]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            parser.load_features_csv(path)

    def test_sort_and_deduplicate(self) -> None:
        frame = make_frame(5)
        shuffled = pd.concat([frame.iloc[[3, 4]], frame.iloc[:3]], ignore_index=True)
        ordered = alignment.sort_by_time(shuffled)
        self.assertTrue(ordered[schema.TIME_COLUMN].is_monotonic_increasing)

        duplicated = pd.concat([ordered, ordered.iloc[[0]]], ignore_index=True)
        deduplicated = alignment.drop_duplicate_timestamps(duplicated)
        self.assertEqual(len(deduplicated), len(ordered))

    def test_alignment_report_finds_gap(self) -> None:
        frame = make_frame(60)
        with_gap = frame.drop(index=range(20, 25)).reset_index(drop=True)
        report = alignment.alignment_report(with_gap, freq_s=1.0)
        self.assertEqual(report.missing_rows, 5)
        self.assertGreaterEqual(report.max_gap_s, 5.0)
        self.assertEqual(report.gap_count, 1)

    def test_missing_timestamps_lists_absent_points(self) -> None:
        frame = make_frame(30)
        with_gap = frame.drop(index=[10, 11]).reset_index(drop=True)
        missing = alignment.missing_timestamps(with_gap)
        self.assertEqual(len(missing), 2)


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class QualityTest(unittest.TestCase):
    """§5.4 Q3 掩码与质量评分。"""

    def test_sat_mask_threshold(self) -> None:
        frame = make_frame(10, cno=0.0)
        self.assertEqual(int(quality.valid_satellite_count(frame).iloc[0]), 0)
        frame_ok = make_frame(10, cno=30.0)
        self.assertEqual(int(quality.valid_satellite_count(frame_ok).iloc[0]), 32)

    def test_attach_masks_columns(self) -> None:
        masked = quality.attach_masks(make_frame(30))
        for column in (quality.VALID_SAT_COUNT_COLUMN, quality.SAT_MASK_COLUMN,
                       quality.GAP_BEFORE_COLUMN, quality.MISS_MASK_COLUMN):
            self.assertIn(column, masked.columns)
        self.assertFalse(bool(masked[quality.MISS_MASK_COLUMN].any()))

    def test_quality_score_bounds_and_breakdown(self) -> None:
        masked = quality.attach_masks(make_frame(120))
        score = quality.data_quality_score(masked)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        breakdown = quality.quality_breakdown(masked)
        self.assertIn("components", breakdown)
        self.assertAlmostEqual(breakdown["components"]["satellite_coverage"], 1.0)


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class FeatureTest(unittest.TestCase):
    """§6.3 派生特征：列齐备且不使用未来行。"""

    def test_signal_features_columns_and_causality(self) -> None:
        masked = quality.attach_masks(make_frame(120, cno=30.0))
        features = signal.signal_features(masked)
        self.assertIn(signal.CN0_VALID_MEAN, features.columns)
        self.assertIn(signal.CN0_DELTA, features.columns)
        # 首行的基线不可用（shift(1)），因此差值为 NaN；其后恒定 C/N0 的差值应为 0
        self.assertTrue(pd.isna(features[signal.CN0_DELTA].iloc[0]))
        self.assertAlmostEqual(float(features[signal.CN0_DELTA].iloc[-1]), 0.0, places=6)

    def test_navigation_clock_difference(self) -> None:
        frame = make_frame(10)
        frame["clkB"] = np.arange(10, dtype="float64") * 2.0
        features = navigation.clock_features(frame)
        self.assertAlmostEqual(float(features["clkB_diff"].iloc[3]), 2.0, places=6)

    def test_observation_outlier_count(self) -> None:
        frame = make_frame(5, residual=1.0)
        frame.loc[2, "Res_G01"] = 100.0
        features = observation.pseudorange_residual(frame, outlier_threshold=15.0)
        self.assertEqual(int(features[observation.RES_OUTLIER_COUNT].iloc[2]), 1)
        self.assertAlmostEqual(float(features[observation.RES_VALID_MAX].iloc[2]), 100.0, places=6)

    def test_invalid_satellite_not_counted(self) -> None:
        """CNO 为 0 的卫星即使残差很大也不应计入（§5.4 Q3）。"""
        frame = make_frame(3, cno=30.0, residual=0.0)
        frame["CNO_G02"] = 0.0
        frame["Res_G02"] = 999.0
        features = observation.pseudorange_residual(frame, outlier_threshold=15.0)
        self.assertEqual(int(features[observation.RES_OUTLIER_COUNT].iloc[0]), 0)


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class ContextAndDetectorTest(unittest.TestCase):
    """§7 Context 与 §8.2 检测器判据。"""

    def _features(self, rows: int = 200) -> "pd.DataFrame":
        masked = quality.attach_masks(make_frame(rows))
        return pd.concat(
            [
                masked,
                signal.signal_features(masked),
                navigation.navigation_features(masked),
                observation.observation_features(masked, outlier_threshold=15.0),
            ],
            axis=1,
        )

    def test_context_components_and_ablation(self) -> None:
        features = self._features()
        encoder = ContextEncoder(enabled=True, history_windows=5)
        context = encoder.encode_from_frame(features, 100)
        self.assertIsNotNone(context)
        # D（检测器历史）由运行期注入，离线编码时为空；is_ready 才是离线可用判据
        self.assertTrue(context.is_ready)
        self.assertFalse(context.is_complete)

        batch = encoder.encode_batch(features.iloc[:20], step=5)
        self.assertEqual(len(batch), 4)
        self.assertIsNone(ContextEncoder(enabled=False).encode_from_frame(features, 0))

    def test_flags_derivation_matches_doc_83_keys(self) -> None:
        thresholds = FlagThresholds()
        flags = derive_flags({"cn0_delta_db": -10.0, "sat_count_delta": -5.0}, thresholds)
        self.assertIn("cn0_decreasing", flags)
        self.assertIn("satellite_abnormal", flags)
        self.assertEqual(derive_flags({}, thresholds), ())

    def test_flags_from_config_are_shared_with_detectors(self) -> None:
        thresholds = FlagThresholds.from_config({"cno": {"cn0_drop_db": -1.5}})
        self.assertAlmostEqual(thresholds.cn0_drop_db, -1.5)

    def test_cno_detector_hit_and_miss(self) -> None:
        detector = CnoDetector()
        config = {"cn0_drop_db": -3.0, "min_valid_sat": 4}
        hit = detector.run({"cn0_delta_db": -8.0, "cn0_valid_count": 10}, None, config)
        self.assertEqual(hit.attack_type.name, "JAMMING")
        self.assertGreater(hit.confidence, 0.9)

        miss = detector.run({"cn0_delta_db": 0.5, "cn0_valid_count": 12}, None, config)
        self.assertEqual(miss.attack_type.name, "NORMAL")

    def test_cno_detector_skips_without_features(self) -> None:
        result = CnoDetector().run({}, None, {})
        self.assertEqual(result.status.value, "skipped")

    def test_satellite_and_pvt_detectors(self) -> None:
        satellite = SatelliteDetector().run({"sat_count_delta": -10.0, "valid_sat_count": 2}, None, {})
        self.assertEqual(satellite.attack_type.name, "JAMMING")

        pvt = PvtDetector().run({"hAcc_vs_baseline": 100.0}, None, {})
        self.assertEqual(pvt.attack_type.name, "SPOOFING")

    def test_observation_detector_reports_prn_evidence(self) -> None:
        # 必须先制造异常、再计算特征，否则特征列仍是异常之前的值
        frame = quality.attach_masks(make_frame(80))
        frame.loc[40, "Res_G03"] = 999.0
        features = pd.concat(
            [
                frame,
                signal.signal_features(frame),
                navigation.navigation_features(frame),
                observation.observation_features(frame, outlier_threshold=15.0),
            ],
            axis=1,
        )
        row = features.iloc[40].to_dict()
        result = ObservationDetector().run(row, None, {"residual_threshold": 15.0, "outlier_count": 1})
        self.assertEqual(result.attack_type.name, "SPOOFING")
        prns = [item["prn"] for item in result.evidence.get("residual_outliers", [])]
        self.assertIn("G03", prns)


@unittest.skipUnless(HAS_PANDAS, "需要 pandas / numpy")
class WindowTest(unittest.TestCase):
    """§6.2 窗口与因果标签。"""

    def test_window_shape_and_causal_label(self) -> None:
        frame = quality.attach_masks(make_frame(50, label=0))
        frame.loc[30:, schema.LABEL_COLUMN] = 2
        X, y, ends = build_windows(frame, [quality.VALID_SAT_COUNT_COLUMN], window_s=10)

        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 1)
        # 窗口末行标签：ends 处的标签与 y 一致，且不早于窗口起点
        for position, end in enumerate(ends):
            self.assertEqual(int(y[position]), int(frame[schema.LABEL_COLUMN].iloc[end]))

    def test_window_rejects_short_data(self) -> None:
        frame = quality.attach_masks(make_frame(5))
        with self.assertRaises(ValueError):
            build_windows(frame, ["cn0_valid_count"], window_s=10)


if __name__ == "__main__":
    unittest.main()
