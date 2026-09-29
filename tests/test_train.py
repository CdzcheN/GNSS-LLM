"""测试 src/train：可复现性、类别权重、划分防泄漏、早停与实验记录。

对应开发文档
    §20.4 固定随机种子、§10.5 类别不平衡、§6.2 窗口配置、§16.2/§16.3 划分与防泄漏、
    §22 早停与过拟合应对、§20.3 实验记录规范。

覆盖要点
    - 种子固定与多 seed 确定性派生；
    - 类别权重口径可用 §5.2 的 1221 计数复算，并与 §10.5 基线值核对；
    - 窗口配置限定在 §6.2 的敏感性集合内；
    - 防泄漏校验对重叠划分硬失败、对随机划分直接拒绝；
    - 早停状态机的改善判定与耐心值；
    - config_hash 的键序无关性与记录字段顺序。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.detectors.base import AttackType
from src.train.dataset import (
    NORMAL_SPLITS,
    WINDOW_CHOICES,
    WindowConfig,
    assert_disjoint,
    assert_no_shuffle,
    expand_day_range,
    loeo_folds,
)
from src.train.experiment import (
    EXPERIMENT_FIELDS,
    ExperimentLog,
    ExperimentRecord,
    config_hash,
    record_from_run,
)
from src.train.losses import (
    BASELINE_CLASS_COUNTS,
    LossSpec,
    balanced_class_weights,
    verify_baseline_weights,
)
from src.train.reproducibility import DEFAULT_SEED, seed_sequence, set_global_seed
from src.train.trainer import EarlyStopping, TrainingConfig


class ReproducibilityTest(unittest.TestCase):
    """§20.4 固定随机种子。"""

    def test_default_seed_is_42(self) -> None:
        self.assertEqual(DEFAULT_SEED, 42)

    def test_set_global_seed_reports_applied_frameworks(self) -> None:
        applied = set_global_seed(7)
        self.assertEqual(applied["python"], 7)
        self.assertIn("numpy", applied)
        self.assertIn("torch", applied)

    def test_seed_sequence_is_deterministic(self) -> None:
        self.assertEqual(seed_sequence(42, 3), (42, 1042, 2042))
        self.assertEqual(seed_sequence(42, 3), seed_sequence(42, 3))

    def test_seed_sequence_rejects_zero_repeats(self) -> None:
        with self.assertRaises(ValueError):
            seed_sequence(42, 0)


class ClassWeightTest(unittest.TestCase):
    """§10.5 类别不平衡。"""

    def test_doc_baseline_weights_are_reproducible_from_counts(self) -> None:
        """用 §5.2 的 1221 计数应能复算出 §10.5 的基线权重。"""
        result = verify_baseline_weights()
        self.assertTrue(result["ok"], result)
        derived = result["derived"]
        self.assertAlmostEqual(derived[int(AttackType.NORMAL)], 1.0, places=3)
        self.assertAlmostEqual(derived[int(AttackType.SPOOFING)], 3.93, places=1)
        self.assertAlmostEqual(derived[int(AttackType.JAMMING)], 70.4, places=0)

    def test_reference_class_is_normalized_to_one(self) -> None:
        weights = balanced_class_weights({0: 100, 1: 50, 2: 10})
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertGreater(weights[2], weights[1])

    def test_invalid_counts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            balanced_class_weights({})
        with self.assertRaises(ValueError):
            balanced_class_weights({0: 0})

    def test_loss_spec_from_config(self) -> None:
        spec = LossSpec.from_config(
            {
                "class_weight": {"Normal": 1.0, "Spoofing": 3.9, "Jamming": 70.0},
                "focal_loss_gamma": 2.0,
                "support": ["resample"],
            }
        )
        self.assertEqual(spec.name, "weighted_ce")
        self.assertEqual(spec.focal_gamma, 2.0)
        self.assertEqual(spec.strategies, ("resample",))
        self.assertAlmostEqual(spec.class_weights[int(AttackType.JAMMING)], 70.0)

    def test_loss_spec_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            LossSpec(name="unknown")
        with self.assertRaises(ValueError):
            LossSpec(strategies=("nope",))

    def test_baseline_counts_match_doc_52(self) -> None:
        self.assertEqual(BASELINE_CLASS_COUNTS[int(AttackType.NORMAL)], 33847)
        self.assertEqual(BASELINE_CLASS_COUNTS[int(AttackType.JAMMING)], 481)


class DatasetProtocolTest(unittest.TestCase):
    """§6.2 窗口与 §16.2/§16.3 划分。"""

    def test_window_choices_match_doc(self) -> None:
        self.assertEqual(WINDOW_CHOICES, (10, 30, 60, 120))

    def test_window_config_default_is_60s(self) -> None:
        config = WindowConfig()
        self.assertEqual(config.window_s, 60)
        self.assertEqual(config.effective_stride_s, 60)

    def test_window_config_rejects_values_outside_sensitivity_set(self) -> None:
        with self.assertRaises(ValueError):
            WindowConfig(window_s=45)

    def test_fixed_window_only_allows_default(self) -> None:
        with self.assertRaises(ValueError):
            WindowConfig(window_s=30, allow_sensitivity=False)

    def test_normal_splits_match_schema(self) -> None:
        """划分以 ``src/data/schema.py:SPLIT_BY_DAY`` 为权威来源。

        当前数据集为 9 月 12–30 日共 19 天，因此 §16.2 的「12–13 / 14 / 15」
        按同一时间连续、互不重叠的原则重划（见 schema 中的说明）。
        """
        self.assertEqual(NORMAL_SPLITS["train"], tuple(range(12, 21)))
        self.assertEqual(NORMAL_SPLITS["validation"], tuple(range(21, 26)))
        self.assertEqual(NORMAL_SPLITS["holdout"], tuple(range(26, 31)))

    def test_expand_day_range_supports_interval_and_enumeration(self) -> None:
        """配置中的日期写法：区间 [12, 20] 与枚举 [12, 13] 都应正确展开。"""
        self.assertEqual(expand_day_range([12, 20]), tuple(range(12, 21)))
        self.assertEqual(expand_day_range([12, 13]), (12, 13))
        self.assertEqual(expand_day_range([14]), (14,))
        with self.assertRaises(ValueError):
            expand_day_range([])

    def test_loeo_folds_match_doc(self) -> None:
        self.assertEqual(loeo_folds(), {"spoofing": 19, "jamming": 10})

    def test_disjoint_passes(self) -> None:
        assert_disjoint({"train": [1, 2], "validation": [3], "holdout": [4]})

    def test_disjoint_raises_on_overlap(self) -> None:
        with self.assertRaises(ValueError):
            assert_disjoint({"train": [1, 2], "holdout": [2, 3]})

    def test_random_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_shuffle("random_state=42")
        assert_no_shuffle(None)


class EarlyStoppingTest(unittest.TestCase):
    """§22 早停状态机。"""

    def test_patience_triggers_stop(self) -> None:
        stopper = EarlyStopping(patience=2, mode="min")
        self.assertFalse(stopper.step(0.5, step=1))
        self.assertFalse(stopper.step(0.6, step=2))
        self.assertTrue(stopper.step(0.7, step=3))
        self.assertTrue(stopper.should_stop)

    def test_improvement_resets_counter(self) -> None:
        stopper = EarlyStopping(patience=2, mode="min")
        stopper.step(0.5, step=1)
        stopper.step(0.6, step=2)
        stopper.step(0.4, step=3)
        self.assertEqual(stopper.counter, 0)
        self.assertEqual(stopper.best_step, 3)
        self.assertEqual(stopper.best, 0.4)

    def test_max_mode(self) -> None:
        stopper = EarlyStopping(patience=1, mode="max")
        stopper.step(0.9, step=1)
        self.assertTrue(stopper.step(0.8, step=2))

    def test_min_delta_is_respected(self) -> None:
        stopper = EarlyStopping(patience=5, mode="min", min_delta=0.1)
        stopper.step(0.5, step=1)
        self.assertFalse(stopper.is_improvement(0.45))

    def test_state_dict_is_serializable(self) -> None:
        stopper = EarlyStopping(patience=1)
        stopper.step(0.5, step=1)
        self.assertIn("best", stopper.state_dict())


class TrainingConfigTest(unittest.TestCase):
    """§10.2 默认配置与取值校验。"""

    def test_defaults_follow_doc(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.seed, DEFAULT_SEED)
        self.assertEqual(config.window_s, 60)
        self.assertEqual(config.num_classes, 3)

    def test_num_classes_must_match_three_state(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(num_classes=2)

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(monitor_mode="sideways")


class ExperimentLogTest(unittest.TestCase):
    """§20.3 实验记录。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_config_hash_is_order_insensitive(self) -> None:
        self.assertEqual(config_hash({"a": 1, "b": 2}), config_hash({"b": 2, "a": 1}))

    def test_config_hash_changes_with_value(self) -> None:
        self.assertNotEqual(config_hash({"a": 1}), config_hash({"a": 2}))

    def test_record_requires_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentRecord(experiment_id="", config_hash="abc")
        with self.assertRaises(ValueError):
            ExperimentRecord(experiment_id="exp-1", config_hash="")

    def test_append_writes_header_in_doc_order(self) -> None:
        log = ExperimentLog(self.tmp / "log.csv")
        record = record_from_run(
            "exp-001", {"seed": 42}, {"macro_f1": 0.9}, figure_paths=["fig1.png"]
        )
        log.append(record)
        header = (self.tmp / "log.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), list(EXPERIMENT_FIELDS))
        self.assertEqual(len(log.read()), 1)

    def test_hash_is_recorded_from_config(self) -> None:
        record = record_from_run("exp-002", {"seed": 42})
        self.assertEqual(record.config_hash, config_hash({"seed": 42}))

    def test_metrics_serialized_as_json(self) -> None:
        log = ExperimentLog(self.tmp / "log.csv")
        row = log.append(record_from_run("exp-003", {"seed": 1}, {"macro_f1": 0.88}))
        self.assertIn("macro_f1", str(row["metrics"]))


if __name__ == "__main__":
    unittest.main()
