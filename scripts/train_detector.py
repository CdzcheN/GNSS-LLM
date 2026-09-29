"""脚本：训练检测器（深度时序模型与基线，§10、§11、§16）。

对应开发文档
    §10 章（模型结构与类别不平衡）、§11 章（无监督 / 半监督辅助路线）、
    §16 章（数据划分与防泄漏）、§20.3 实验记录规范、§20.4 固定随机种子、§22 风险管理。

用法
    # 查看训练计划（不实际训练）
    python -m scripts.train_detector --dry-run

    # 正式训练（需要 PyTorch 与已生成的特征表）
    python -m scripts.train_detector --features data/features_1221.csv \
        --model lstm --loss weighted_ce --seed 42 --epochs 30 \
        --out results/checkpoints/lstm.pt

流程
    1. 读取特征表并按 §16.2 划分（时间连续、不重叠，禁止随机打散）
    2. 构造窗口张量 B × W × D（§6.2）
    3. 训练：加权 CE / Focal Loss（§10.5），按验证集指标早停（§22）
    4. 保存 checkpoint，并把一行实验记录写入 results/experiments_log.csv（§20.3）

依赖说明
    - 训练需要 PyTorch（requirements.txt 中的 ``torch``）；未安装时脚本会给出明确提示；
    - 无监督预训练（§11）需额外流程，见 ``src/train/unsupervised.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # 支持 `python scripts/train_detector.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import schema  # noqa: E402
from src.detectors.deep_temporal import (  # noqa: E402
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_WINDOW_S,
    PARAMETER_BUDGET,
)
from src.train.experiment import ExperimentLog, record_from_run  # noqa: E402
from src.train.losses import IMBALANCE_STRATEGIES, LOSS_NAMES  # noqa: E402
from src.train.reproducibility import DEFAULT_SEED, describe_environment, seed_sequence  # noqa: E402
from src.train.trainer import TrainingConfig  # noqa: E402

#: 支持的模型骨干（与 config.yaml 的 ``model.baselines`` 保持一致，§10.2、§10.4）。
MODEL_CHOICES: tuple[str, ...] = ("lstm", "temporal_cnn", "small_transformer")

#: 支持的训练路线（§11）。
ROUTE_CHOICES: tuple[str, ...] = ("supervised", "unsupervised", "semisupervised")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="训练 GNSS 检测器（开发文档 §10、§11、§16）")
    ap.add_argument("--features", default="data/features_1221.csv",
                    help="特征表（由 scripts/extract_features.py 生成）")
    ap.add_argument("--model", default="lstm", choices=MODEL_CHOICES,
                    help="模型骨干（主模型为 LSTM，§10.2；基线见 §10.4）")
    ap.add_argument("--loss", default="weighted_ce", choices=LOSS_NAMES, help="损失函数（§10.5）")
    ap.add_argument("--route", default="supervised", choices=ROUTE_CHOICES, help="训练路线（§11）")
    ap.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S, help="窗口长度（§6.2）")
    ap.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="隐藏维度（§10.2）")
    ap.add_argument("--lstm-layers", type=int, default=DEFAULT_LSTM_LAYERS, help="LSTM 层数（§10.2）")
    ap.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA, help="Focal γ（§10.5）")
    ap.add_argument("--epochs", type=int, default=30, help="最大训练轮数")
    ap.add_argument("--batch-size", type=int, default=64, help="批大小")
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    ap.add_argument("--patience", type=int, default=5, help="早停耐心值（§22）")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（§20.4 基线 42）")
    ap.add_argument("--seeds", type=int, default=1, help="重复训练的次数（§20.4 多 seed）")
    ap.add_argument("--out", default="results/checkpoints/model.pt", help="checkpoint 输出路径")
    ap.add_argument("--log", default="results/experiments_log.csv", help="实验记录文件（§20.3）")
    ap.add_argument("--experiment-id", default=None, help="实验编号；缺省时按种子自动生成")
    ap.add_argument("--dry-run", action="store_true", help="只打印训练计划，不执行训练")
    return ap


def _require_torch() -> Any:
    """惰性导入 PyTorch。

    Raises:
        ImportError: 未安装 torch。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "训练需要 PyTorch，请执行 `pip install -r requirements.txt`"
            "（CPU 版可用 --index-url https://download.pytorch.org/whl/cpu）"
        ) from exc
    return torch


def build_training_config(args: argparse.Namespace, seed: int) -> TrainingConfig:
    """由命令行参数构造训练配置。

    Args:
        args: 解析后的参数。
        seed: 本次运行的随机种子。

    Returns:
        TrainingConfig。
    """
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stop_patience=args.patience,
        seed=seed,
        window_s=args.window_s,
    )


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码（0 成功；1 缺依赖或文件）。

    Raises:
        SystemExit: 参数解析失败。
    """
    args = build_parser().parse_args(argv)

    seeds = seed_sequence(args.seed, max(1, args.seeds))
    print(f"[train_detector] 骨干={args.model} 损失={args.loss} 路线={args.route}")
    print(f"[train_detector] LSTM {args.lstm_layers}×{args.hidden_size}，窗口 {args.window_s}s，"
          f"类别数 {len(schema.LABEL_NAMES)}")
    print(f"[train_detector] 类别权重基线：{ {schema.LABEL_NAMES[k]: v for k, v in DEFAULT_CLASS_WEIGHTS.items()} }")
    print(f"[train_detector] 参数量目标 < {PARAMETER_BUDGET}")
    print(f"[train_detector] 环境：{describe_environment()}")
    print(f"[train_detector] 种子序列：{seeds}")
    print(f"[train_detector] 特征表：{args.features}")

    if args.dry_run:
        for seed in seeds:
            config = build_training_config(args, seed)
            print(f"[train_detector] （dry-run）seed={config.seed} epochs={config.epochs} "
                  f"batch={config.batch_size} lr={config.learning_rate} patience={config.early_stop_patience}")
        print("[train_detector] dry-run 结束：未执行训练")
        return 0

    features_path = Path(args.features)
    if not features_path.exists():
        print(f"[train_detector] 特征表不存在：{features_path}")
        print("[train_detector] 请先运行：python -m scripts.extract_features --dataset three_state "
              f"--out {args.features}")
        return 1

    _require_torch()

    from src.detectors.deep_temporal import DeepTemporalDetector  # 局部导入，避免无 torch 时也要解析

    log = ExperimentLog(args.log)
    detector = DeepTemporalDetector(
        num_layers=args.lstm_layers,
        hidden_size=args.hidden_size,
        window_s=args.window_s,
    )

    for seed in seeds:
        config = build_training_config(args, seed)
        experiment_id = args.experiment_id or f"{args.model}-seed{seed}"
        print(f"[train_detector] 开始训练 {experiment_id} …")
        try:
            result = detector.train(  # type: ignore[attr-defined]
                features_path=str(features_path),
                config=config,
                loss=args.loss,
                checkpoint_path=args.out,
            )
        except NotImplementedError as exc:
            print(f"[train_detector] 训练流程尚未实现：{exc}")
            return 1

        log.append(
            record_from_run(
                experiment_id=experiment_id,
                config=vars(args),
                metrics={"best_metric": result.best_metric, "epochs_run": result.epochs_run},
                seed=seed,
                model_version=detector.version,
                data_version=features_path.name,
                split_version=str(schema.SPLIT_BY_DAY),
                checkpoint_path=args.out,
            )
        )
        print(f"[train_detector] {experiment_id} 完成：best={result.best_metric} "
              f"epochs={result.epochs_run} 早停={result.stopped_early}")

    print(f"[train_detector] 实验记录：{args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
