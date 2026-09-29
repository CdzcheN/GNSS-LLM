"""脚本：一键端到端检查 —— 数据 → 特征 → Context → 检测 → 融合 → 事件 → 指标。

对应开发文档
    §6.1 处理流水线、§7 Context、§9 策略选择、§12 融合、§13 事件、§15 实时流程、§17 指标。

用法
    # 用三态数据前 5000 行跑通全链路（默认）
    python -m scripts.run_pipeline_check

    # 调大样本量（例如 50000 行）观察稳定性
    python -m scripts.run_pipeline_check --nrows 50000

    # 用正常数据（前 5000 行）验证“无异常时不应产生事件”
    python -m scripts.run_pipeline_check --dataset normal --nrows 5000

    # 指定输出目录（中间产物与结果）
    python -m scripts.run_pipeline_check --workdir results/check

检查内容
    1. 数据集文件是否齐备、行数是否与 schema 记录一致；
    2. 扩展特征表（掩码 + 信号/导航/观测特征）能否成功构建；
    3. Context 能否编码，六分量与置信度是否合理；
    4. 闭环（策略 → 检测 → 融合 → 事件）能否运行，统计是否正确；
    5. 逐历元指标（Accuracy / Macro-F1，Jamming 单列）与事件级指标。

退出码
    0 全部通过；1 数据缺失或任一环节失败（消息中给出原因）。

注意
    这是**链路自检**，不替代 §17/§18 的正式评估实验；检测器阈值尚未标定（见 config.yaml），
    因此指标仅供“链路是否通畅”的判断，不能作为论文结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # 支持 `python scripts/run_pipeline_check.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import parser, schema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="端到端链路自检（开发文档 §6–§17）")
    ap.add_argument("--dataset", default="three_state", choices=["three_state", "normal"],
                    help="使用三态数据或正常数据")
    ap.add_argument("--nrows", type=int, default=5000, help="样本行数")
    ap.add_argument("--workdir", default="results/check", help="中间产物与结果输出目录")
    ap.add_argument("--policy", default="rule", choices=["rule", "fixed"], help="策略层级（§9.6）")
    ap.add_argument("--step", type=int, default=1, help="闭环推进步长")
    ap.add_argument("--config", default="config.yaml", help="配置文件")
    return ap


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("[check] 需要 pandas：pip install -r requirements.txt")
        return 1

    # ---- 1. 数据集是否齐备 -------------------------------------------------
    print("[check] 1/5 数据集概况")
    overview = parser.describe_dataset()
    missing = [name for name, info in overview.items() if not info["exists"]]
    for name, info in overview.items():
        print(f"    {'存在' if info['exists'] else '缺失'}  {name}  {info['size_mb']} MB")
    if missing:
        print(f"[check] 失败：缺少数据文件 {missing}（请放入 data/ 目录）")
        return 1
    print(f"    schema：{schema.EXPECTED_COLUMN_COUNT} 列 / {schema.EXPECTED_FEATURE_COUNT} 数值特征")

    # ---- 2. 特征扩展 -------------------------------------------------------
    print(f"[check] 2/5 构建特征表（{args.dataset}，前 {args.nrows} 行）")
    from scripts.extract_features import extract_features  # 复用同一实现，避免两套逻辑

    name = schema.THREE_STATE_FILE if args.dataset == "three_state" else schema.NORMAL_FILES[0]
    try:
        features = extract_features(name, args.nrows, residual_threshold=15.0)
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要给出可读失败原因
        print(f"[check] 失败：特征构建异常 {type(exc).__name__}: {exc}")
        return 1
    print(f"    特征表：{features.shape[0]} 行 × {features.shape[1]} 列")
    if args.dataset == "three_state":
        counts = features[schema.LABEL_COLUMN].value_counts().to_dict()
        print(f"    标签分布：{ {schema.LABEL_NAMES.get(int(k), k): int(v) for k, v in counts.items()} }")

    # ---- 3. Context 编码 ---------------------------------------------------
    print("[check] 3/5 构建 Context")
    from src.context.context_encoder import ContextEncoder, FlagThresholds

    config = _load_config(args.config)
    encoder = ContextEncoder(
        enabled=True,
        thresholds=FlagThresholds.from_config(config.get("detectors")),
    )
    contexts = encoder.encode_batch(features, step=max(1, args.nrows // 100))
    filled = [c for c in contexts if c is not None and c.is_ready]
    print(f"    编码 {len(contexts)} 个 Context，其中六分量齐备 {len(filled)} 个")
    if contexts and contexts[0] is not None:
        print(f"    首条 Context：confidence={contexts[0].confidence:.3f} "
              f"flags={contexts[0].flags}")

    # ---- 4. 闭环运行 -------------------------------------------------------
    print(f"[check] 4/5 运行闭环（策略={args.policy}）")
    from scripts.run_agent import DEFAULT_PREDICTION_FEATURES, build_registry, run_closed_loop
    from src.agent.policy import ContextRulePolicy, FixedPolicy
    from src.event.event_logger import EventLogger
    from src.event.event_manager import EventManager

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    registry = build_registry()
    policy = FixedPolicy() if args.policy == "fixed" else ContextRulePolicy()
    event_config = config.get("event") or {}
    manager = EventManager(
        merge_window_s=float(event_config.get("alert_merge_window_s", 30.0)),
        confirm_after_s=float(event_config.get("confirm_after_s", 0.0)),
    )
    logger = EventLogger(workdir / "events.jsonl", workdir / "events.csv")

    try:
        stats = run_closed_loop(
            frame=features,
            registry=registry,
            policy=policy,
            encoder=encoder,
            manager=manager,
            logger=logger,
            detector_configs=dict(config.get("detectors") or {}),
            step=args.step,
        )
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要给出可读失败原因
        print(f"[check] 失败：闭环运行异常 {type(exc).__name__}: {exc}")
        return 1

    print(f"    处理历元：{stats['epochs']}，形成事件：{stats['events']}")
    print(f"    平均检测器调用数：{stats['avg_detector_calls']:.2f}，"
          f"冲突率：{stats['conflict_rate']:.4f}")
    print(f"    事件日志：{workdir / 'events.jsonl'}")

    # ---- 5. 指标 -----------------------------------------------------------
    print("[check] 5/5 计算指标（§17）")
    predictions = pd.DataFrame(stats["predictions"])
    prediction_path = workdir / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    from src.eval.metrics import classification_report

    labels = tuple(sorted(schema.LABEL_NAMES))
    report = classification_report(
        predictions[schema.LABEL_COLUMN].tolist(),
        predictions["prediction"].tolist(),
        labels=labels,
    )
    print(f"    Accuracy={report['accuracy']:.4f}  Macro-F1={report['macro_f1']:.4f}")
    for label_name, values in (report.get("per_class") or {}).items():
        print(f"      {label_name:9s} P={values['precision']:.3f} R={values['recall']:.3f} "
              f"F1={values['f1']:.3f} support={int(values['support'])}")
    print(f"    预测文件：{prediction_path}")

    # ---- 6. 结果可视化（§17、§20.3 的 figure_paths）-------------------------
    print("[check] 6/6 生成结果图件")
    try:
        from src.eval.plots import make_evaluation_figures

        figures = make_evaluation_figures(
            predictions,
            figures_dir=workdir / "figures",
            prefix=f"{args.dataset}_check",
            feature_columns=DEFAULT_PREDICTION_FEATURES,
        )
        print(f"    已生成 {len(figures)} 张图：")
        for item in figures:
            print(f"      {item}")
        print("    看图要点：timeline 看错误集中在哪些时段；confusion 看谁被误判成谁；")
        print("              features 看真实异常区间内特征是否真的发生变化（判断可分性）。")
    except Exception as exc:  # noqa: BLE001 - 出图失败不应影响链路自检结论
        print(f"    出图失败（不影响链路结论）：{type(exc).__name__}: {exc}")

    print("[check] 完成：链路通畅。")
    print("[check] 注意：S3/S5/S6 的阈值已在验证集上初步标定（见 config.yaml 内注释），"
          "但尚未做事件级交叉验证（LOEO），指标仍不代表最终性能。")
    print("[check] 重新标定请运行：python -m scripts.calibrate_thresholds --emit-yaml")
    return 0


def _load_config(path: str) -> dict:
    """读取 YAML 配置；缺失时返回空配置（不阻断自检）。"""
    target = Path(path)
    if not target.exists():
        print(f"[check] 提示：未找到 {path}，改用内置默认阈值")
        return {}
    try:
        import yaml
    except ImportError:
        print("[check] 提示：未安装 PyYAML，改用内置默认阈值")
        return {}
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    raise SystemExit(main())
