"""脚本：评估与消融实验，输出五层指标与消融计划（§17、§18）。

对应开发文档
    §17 评价指标体系（逐秒 / 类别 / 事件 / 策略 / 系统五层）、§18 消融与对比实验、
    §16.3 防泄漏规范、§20.3 实验记录规范。

用法
    # 由预测结果 CSV 计算逐秒与类别级指标（Jamming 单独报告，§17.2）
    python -m scripts.evaluate --predictions results/pred_1221.csv \
        --true-column Label --pred-column prediction --out results/metrics/eval_1221.json

    # 列出 §18 的八个消融项及其代码开关
    python -m scripts.evaluate --list-ablations

    # 查看某个消融项的对照组
    python -m scripts.evaluate --ablation modality

流程
    1. 读取预测结果（真实列与预测列可配置）
    2. 计算混淆矩阵与各类 P/R/F1、Macro-F1（§17.1、§17.2）
    3. 若提供事件区间列，则计算事件级指标（§17.3）
    4. 写出指标 JSON 与一行实验记录（§20.3）

关键约束
    - 指标只能在同一划分内计算（§16.3）；本脚本不合并不同划分的数据；
    - Jamming 必须单独报告，不允许只给 Macro 值（§17.2）；
    - 结果优先于指标：不得用单一 Accuracy 代表整体效果（§2.6）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):  # 支持 `python scripts/evaluate.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import schema  # noqa: E402
from src.eval.ablation import ABLATION_KEYS, get_ablation, summary, validate_arms  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    CLASS_LEVEL_METRICS,
    EVENT_LEVEL_METRICS,
    SECOND_LEVEL_METRICS,
    STRATEGY_LEVEL_METRICS,
    SYSTEM_LEVEL_METRICS,
    Interval,
    classification_report,
    event_level_metrics,
)
from src.train.experiment import EXPERIMENT_FIELDS, ExperimentLog, record_from_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="评估与消融实验（开发文档 §17、§18）")
    ap.add_argument("--predictions", help="预测结果 CSV（含真实列与预测列）")
    ap.add_argument("--true-column", default=schema.LABEL_COLUMN, help="真实标签列名")
    ap.add_argument("--pred-column", default="prediction", help="预测标签列名")
    ap.add_argument("--nrows", type=int, default=None, help="读取行数上限")
    ap.add_argument("--split", default="holdout", choices=["train", "validation", "holdout", "loeo"],
                    help="本结果所属划分（§16.2；仅用于记录，不改变计算）")
    ap.add_argument("--out", default=None, help="指标 JSON 输出路径")
    ap.add_argument("--log", default="results/experiments_log.csv", help="实验记录文件（§20.3）")
    ap.add_argument("--experiment-id", default=None, help="实验编号")
    ap.add_argument("--figures-dir", default="results/figures",
                    help="图件输出目录（§20.3 的 figure_paths）；传空字符串表示不出图")
    ap.add_argument("--feature-columns", default="cn0_delta_db,res_valid_mean,pDOP_vs_baseline,valid_sat_count",
                    help="需要绘制曲线对比的特征列（逗号分隔）")
    ap.add_argument("--ablation", choices=ABLATION_KEYS, help="查看某个消融项定义（§18）")
    ap.add_argument("--arms", default=None, help="消融对照组（逗号分隔），配合 --ablation 使用")
    ap.add_argument("--list-ablations", action="store_true", help="列出全部消融项")
    return ap


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def print_metric_catalog() -> None:
    """打印 §17 的五层指标清单。"""
    print("[evaluate] §17 五层指标：")
    for name, keys in (
        ("逐秒级", SECOND_LEVEL_METRICS),
        ("类别级", CLASS_LEVEL_METRICS),
        ("事件级", EVENT_LEVEL_METRICS),
        ("策略层", STRATEGY_LEVEL_METRICS),
        ("系统级", SYSTEM_LEVEL_METRICS),
    ):
        print(f"  {name}（{len(keys)}）：{list(keys)}")


def evaluate_predictions(
    frame: Any,
    true_column: str,
    pred_column: str,
) -> Mapping[str, Any]:
    """由预测结果表计算指标。

    Args:
        frame: 含真实列与预测列的 DataFrame（须保持时间顺序）。
        true_column: 真实标签列。
        pred_column: 预测标签列。

    Returns:
        含 ``classification`` 与（若有事件列）``events`` 的指标映射。

    Raises:
        ValueError: 缺少列。
    """
    for column in (true_column, pred_column):
        if column not in frame.columns:
            raise ValueError(f"预测文件缺少列 {column!r}（现有列：{list(frame.columns)[:8]}…）")

    labels = tuple(sorted(set(schema.LABEL_NAMES)))
    report = classification_report(
        frame[true_column].tolist(), frame[pred_column].tolist(), labels=labels
    )
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "classification": dict(report),
        "label_distribution": {
            schema.LABEL_NAMES.get(int(label), str(label)): int(count)
            for label, count in frame[true_column].value_counts().items()
        },
    }

    event_columns = {"start_time", "end_time"}
    if event_columns.issubset(frame.columns):
        predicted = [
            Interval(float(row["start_time"]), float(row["end_time"]))
            for _, row in frame.iterrows()
            if row.get(pred_column) is not None
        ]
        truth = [
            Interval(float(row["start_time"]), float(row["end_time"]))
            for _, row in frame.iterrows()
        ]
        metrics["events"] = dict(event_level_metrics(predicted, truth))
    return metrics


def print_report(metrics: Mapping[str, Any]) -> None:
    """以人读形式打印指标（Jamming 单独一行，§17.2）。"""
    classification = metrics.get("classification", {})
    print(f"[evaluate] 样本数：{metrics.get('rows')}")
    print(f"[evaluate] Accuracy = {classification.get('accuracy'):.4f}  "
          f"Macro-F1 = {classification.get('macro_f1'):.4f}")
    for name, values in (classification.get("per_class") or {}).items():
        if isinstance(values, Mapping):
            print(f"[evaluate]   {name:9s} P={values.get('precision'):.4f} "
                  f"R={values.get('recall'):.4f} F1={values.get('f1'):.4f} "
                  f"support={int(values.get('support', 0))}")
    print(f"[evaluate] 标签分布：{metrics.get('label_distribution')}")
    if "events" in metrics:
        print(f"[evaluate] 事件级：{metrics['events']}")


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    if args.list_ablations:
        print("[evaluate] §18 消融项：")
        for entry in summary():
            print(f"  {entry['key']:18s} {entry['section']}  对照组={entry['arms']}")
            print(f"     开关：{entry['switch']}")
        return 0

    if args.ablation:
        arms = [item.strip() for item in (args.arms or "").split(",") if item.strip()]
        chosen = validate_arms(args.ablation, arms)
        spec = get_ablation(args.ablation)
        print(f"[evaluate] {spec.section} {spec.key}：对照组={list(chosen)}")
        print(f"[evaluate] 开关：{spec.switch}")
        if spec.note:
            print(f"[evaluate] 说明：{spec.note}")
        return 0

    print_metric_catalog()

    if not args.predictions:
        print("[evaluate] 未提供 --predictions：仅打印指标口径与消融清单。")
        print("[evaluate] 提示：先用 scripts/run_agent.py 产出预测，再回到本脚本评估。")
        return 0

    path = Path(args.predictions)
    if not path.exists():
        print(f"[evaluate] 预测文件不存在：{path}")
        return 1

    pd = _require_pandas()
    frame = pd.read_csv(path, nrows=args.nrows)
    metrics = evaluate_predictions(frame, args.true_column, args.pred_column)
    print_report(metrics)

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[evaluate] 指标已写出：{target}")

    figure_paths: list[str] = []
    if args.figures_dir:
        try:
            from src.eval.plots import make_evaluation_figures

            feature_columns = [name.strip() for name in (args.feature_columns or "").split(",") if name.strip()]
            figures = make_evaluation_figures(
                frame,
                figures_dir=args.figures_dir,
                prefix=f"{args.split}_{path.stem}",
                feature_columns=feature_columns,
                true_column=args.true_column,
                pred_column=args.pred_column,
            )
            figure_paths = [str(item) for item in figures]
            print(f"[evaluate] 已生成图件 {len(figures)} 张：")
            for item in figures:
                print(f"    {item}")
        except Exception as exc:  # noqa: BLE001 - 出图失败不应影响指标结论
            print(f"[evaluate] 出图失败（指标不受影响）：{type(exc).__name__}: {exc}")

    print("[evaluate] 怎么看这些图：")
    print("    *_timeline.png    上=真实、中=预测、下=预测错误时段，错误集中在哪一目了然")
    print("    *_confusion.png   谁被误判成谁（行占比对小样本的 Jamming 尤其重要）")
    print("    *_class_metrics.png  各类 P/R/F1，可看出 Jamming 是否被完全漏检")
    print("    *_features.png    关键特征曲线 + 真实异常区间底色；若异常区间内特征无明显变化，")
    print("                      说明该特征无区分度，问题不在检测阈值上")

    experiment_id = args.experiment_id or f"eval-{args.split}-{path.stem}"
    ExperimentLog(args.log).append(
        record_from_run(
            experiment_id=experiment_id,
            config={"predictions": str(path), "split": args.split},
            metrics={"accuracy": metrics["classification"].get("accuracy"),
                     "macro_f1": metrics["classification"].get("macro_f1")},
            split_version=f"{args.split}:{schema.SPLIT_BY_DAY.get(args.split, 'n/a')}",
            conclusion="",
            figure_paths=figure_paths,
        )
    )
    print(f"[evaluate] 实验记录已追加：{args.log}（字段 {list(EXPERIMENT_FIELDS)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
