"""测试 config.yaml 与代码默认常量的一致性（防止配置与实现漂移）。

对应开发文档
    §20.1 技术栈（YAML 配置）、§20.3 实验记录规范、§28 文档维护规则、
    §6.2 时间窗口、§5.4 Q3 卫星掩码、§10.2/§10.5 模型与类别不平衡、
    §13.2/§13.3 事件、§17 评价指标、§8.4 注册清单。

职责
    把「配置项」与「代码默认值」的对应关系固化为断言，避免两处独立演化后对不上：
        - 数据：窗口长度、sat_mask 阈值；
        - 模型：LSTM 层数 / hidden / 类别数 / 骨干；
        - 类别不平衡：类别权重与 focal γ；
        - 事件：告警合并窗口与状态集合；
        - 评价指标：五层指标键；
        - 检测器注册清单与脚本默认值。

不做（边界）
    - 不判断配置项的取值是否“最优”（那是实验的任务）；
    - 不测试 config.yaml 中尚未有代码实现的占位项（如 splits 的日期划分）。
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from scripts import evaluate as evaluate_script
from scripts import run_agent as run_agent_script
from scripts import train_detector as train_script
from src.data.quality import SAT_MASK_CNO_THRESHOLD
from src.detectors.base import AttackType
from src.detectors.deep_temporal import (
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW_S,
)
from src.event.event_manager import DEFAULT_ALERT_MERGE_WINDOW_S
from src.event.event_record import EventState

#: 项目根目录与配置文件路径。
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH: pathlib.Path = ROOT / "config.yaml"


class ConfigConsistencyTest(unittest.TestCase):
    """config.yaml 与代码常量逐项对账。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 数据与模型

    def test_data_section_matches_code(self) -> None:
        """§6.2 窗口与 §5.4 Q3 掩码阈值。"""
        data = self.config["data"]
        self.assertEqual(data["window_s"], DEFAULT_WINDOW_S)
        self.assertEqual(data["sat_mask_cno_threshold"], SAT_MASK_CNO_THRESHOLD)
        self.assertEqual(data["labels_csv"], "labels_1221.csv")

    def test_model_section_matches_code(self) -> None:
        """§10.2 配置基线。"""
        model = self.config["model"]
        self.assertEqual(model["lstm_layers"], DEFAULT_LSTM_LAYERS)
        self.assertEqual(model["hidden_size"], DEFAULT_HIDDEN_SIZE)
        self.assertEqual(model["num_classes"], DEFAULT_NUM_CLASSES)
        self.assertEqual(model["window_s"], DEFAULT_WINDOW_S)
        self.assertEqual(sorted(model["labels"].values()), ["Jamming", "Normal", "Spoofing"])

    def test_class_imbalance_matches_code(self) -> None:
        """§10.5 类别权重与 focal γ。"""
        imbalance = self.config["class_imbalance"]
        self.assertEqual(imbalance["focal_loss_gamma"], DEFAULT_FOCAL_GAMMA)
        weights = imbalance["class_weight"]
        self.assertEqual(weights["Normal"], DEFAULT_CLASS_WEIGHTS[int(AttackType.NORMAL)])
        self.assertEqual(weights["Spoofing"], DEFAULT_CLASS_WEIGHTS[int(AttackType.SPOOFING)])
        self.assertEqual(weights["Jamming"], DEFAULT_CLASS_WEIGHTS[int(AttackType.JAMMING)])

    def test_model_baselines_supported_by_train_script(self) -> None:
        """§10.4 基线必须被训练脚本的 --model 接受。"""
        for baseline in self.config["model"]["baselines"]:
            self.assertIn(baseline, train_script.MODEL_CHOICES)
        self.assertEqual(self.config["model"]["backbone"], "lstm")

    def test_seed_matches_train_script_default(self) -> None:
        """§20.4 固定随机种子。"""
        parser = train_script.build_parser()
        self.assertEqual(parser.get_default("seed"), self.config["project"]["seed"])

    # ---------------------------------------------------------------- 事件

    def test_event_section_matches_code(self) -> None:
        """§13.2 合并窗口与 §13.3 状态集合。"""
        event = self.config["event"]
        self.assertEqual(event["alert_merge_window_s"], DEFAULT_ALERT_MERGE_WINDOW_S)
        self.assertEqual(
            tuple(state.lower() for state in event["states"]),
            tuple(state.value for state in EventState),
        )

    # ---------------------------------------------------------------- 注册表与指标

    def test_registered_detectors_match_run_agent(self) -> None:
        """config 的注册清单必须与 run_agent.build_registry() 实际注册的一致（§8.2、§8.4）。"""
        registry = run_agent_script.build_registry()
        self.assertEqual(sorted(registry.ids()), sorted(self.config["detectors"]["registered"]))

    def test_evaluation_metrics_match_evaluate_script(self) -> None:
        """§17 五层指标键必须与 scripts/evaluate.py 的常量完全一致。"""
        evaluation = self.config["evaluation"]
        self.assertEqual(tuple(evaluation["second_level"]), evaluate_script.SECOND_LEVEL_METRICS)
        self.assertEqual(tuple(evaluation["class_level"]), evaluate_script.CLASS_LEVEL_METRICS)
        self.assertEqual(tuple(evaluation["event_level"]), evaluate_script.EVENT_LEVEL_METRICS)
        self.assertEqual(tuple(evaluation["strategy_level"]), evaluate_script.STRATEGY_LEVEL_METRICS)
        self.assertEqual(tuple(evaluation["system_level"]), evaluate_script.SYSTEM_LEVEL_METRICS)


if __name__ == "__main__":
    unittest.main()
