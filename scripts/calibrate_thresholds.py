"""脚本：在验证集上为各检测器判据标定阈值（§8.2、§16.3、§20.3）。

对应开发文档
    §8.2 策略分类与判据、§8.3 策略适用性、§16.2 数据划分、§16.3 防泄漏规范、
    §17.1 指标（AUC-ROC / P / R / F1）、§20.3 实验记录。

用法
    # 直接从 data/ 的原始三态数据现算特征并标定（推荐，无需先跑 extract_features）
    python -m scripts.calibrate_thresholds

    # 只看某个检测器 / 某个目标类别
    python -m scripts.calibrate_thresholds --detector observation --target spoofing

    # 直接打印可粘贴到 config.yaml 的片段
    python -m scripts.calibrate_thresholds --emit-yaml

    # 使用已生成的特征表（含派生特征列）
    python -m scripts.calibrate_thresholds --features data/features_1221.csv

流程
    1. 读特征表 → 按 §16.2 **事件级切分**（保证验证集含所有类别，避免“验证集全 Normal”）；
    2. 对每个检测器的候选判据特征，在**验证集**上扫描阈值：
       方向（``<=`` / ``>=``）× 数据分位点 → TP/FP/TN/FN → Precision / Recall / F1；
    3. 取 F1 最大者为建议工作点，同时给出该特征的 AUC（判断是否有可用排序能力）；
    4. 输出建议值与可粘贴的 ``config.yaml`` 片段。

关键约束
    - **只用验证集标定**，绝不用测试 / 留出集（§16.3）；
    - 阈值候选由数据分位数生成，不凭经验硬编码；
    - 每组都报告 AUC：**AUC ≈ 0.5 时调阈值无意义**，应回到特征层（§6.3）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # 支持 `python scripts/calibrate_thresholds.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import schema  # noqa: E402
from src.train.dataset import split_by_events  # noqa: E402

#: 各检测器的标定计划：候选判据特征 → 对应的 config.yaml 键名。
CALIBRATION_PLAN: Mapping[str, Mapping[str, Any]] = {
    "cno": {
        "attack": "jamming",
        "features": ("cn0_delta_db", "cn0_valid_count", "cn0_valid_mean", "AvgCNO"),
        "config_keys": {
            "cn0_delta_db": "cn0_drop_db",
            "cn0_valid_count": "min_valid_sat",
        },
    },
    "satellite": {
        "attack": "jamming",
        "features": ("sat_count_delta", "valid_sat_count", "NumSats"),
        "config_keys": {
            "sat_count_delta": "sat_count_drop",
            "valid_sat_count": "min_sat_count",
        },
    },
    "observation": {
        "attack": "spoofing",
        "features": ("MaxRes", "res_valid_max", "res_valid_mean", "res_outlier_count", "res_delta"),
        "config_keys": {
            "MaxRes": "residual_threshold",
            "res_valid_max": "residual_threshold",
            "res_outlier_count": "outlier_count",
            "res_delta": "baseline_rise",
        },
    },
    "pvt": {
        "attack": "spoofing",
        "features": ("hAcc_vs_baseline", "vAcc_vs_baseline", "pDOP_vs_baseline", "clkB_diff"),
        "config_keys": {
            "hAcc_vs_baseline": "acc_rise",
            "vAcc_vs_baseline": "acc_rise",
            "pDOP_vs_baseline": "dop_rise",
            "clkB_diff": "clock_jump",
        },
    },
}

#: 目标类别名 → 标签值。
TARGET_LABELS: Mapping[str, int] = {"spoofing": 1, "jamming": 2}


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="在验证集上标定检测器阈值（开发文档 §8.2、§16.3）")
    ap.add_argument("--features", help="已生成的特征表（含派生列）；缺省则从原始数据现算")
    ap.add_argument("--nrows", type=int, default=None, help="读取行数上限（快速试验用）")
    ap.add_argument("--residual-threshold", type=float, default=15.0,
                    help="生成 res_outlier_count 时使用的残差阈值")
    ap.add_argument("--detector", choices=sorted(CALIBRATION_PLAN), help="只标定指定检测器")
    ap.add_argument("--target", choices=sorted(TARGET_LABELS), help="只标定指定目标类别")
    ap.add_argument("--quantiles", type=int, default=200, help="阈值候选个数（按数据分位点）")
    ap.add_argument("--validation-event-ratio", type=float, default=1.0 / 3.0,
                    help="异常类别用于验证的事件段比例（§16.2 事件级切分）")
    ap.add_argument("--emit-yaml", action="store_true", help="输出可粘贴到 config.yaml 的片段")
    ap.add_argument("--top", type=int, default=5, help="每个检测器展示前 N 个特征")
    return ap


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def load_features(args: argparse.Namespace) -> Any:
    """载入特征表（优先读文件，否则现算）。

    Args:
        args: 命令行参数。

    Returns:
        含标签与派生特征列的 DataFrame。

    Raises:
        FileNotFoundError: 指定了不存在的特征表。
        ImportError: 未安装 pandas。
    """
    pd = _require_pandas()
    if args.features:
        path = Path(args.features)
        if not path.exists():
            raise FileNotFoundError(f"特征表不存在：{path}")
        frame = pd.read_csv(path, nrows=args.nrows)
        print(f"[calibrate] 读取特征表：{path}（{len(frame)} 行）")
        return frame

    from scripts.extract_features import extract_features

    frame = extract_features(schema.THREE_STATE_FILE, args.nrows, args.residual_threshold)
    print(f"[calibrate] 由原始数据现算特征：{len(frame)} 行 × {frame.shape[1]} 列")
    return frame


def auc_score(values: Any, positive: Any, direction: str) -> float:
    """计算单特征的 ROC AUC（秩方法，不依赖 scikit-learn）。

    Args:
        values: 特征值数组。
        positive: 二值标签（1 = 目标类别）。
        direction: ``"le"``（越小越可能是目标）或 ``"ge"``（越大越可能）。

    Returns:
        AUC；某一类缺失时返回 ``nan``。
    """
    import numpy as np

    score = -np.asarray(values, dtype="float64") if direction == "le" else np.asarray(values, dtype="float64")
    labels = np.asarray(positive).astype(bool)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype="float64")
    ranks[order] = np.arange(1, len(score) + 1, dtype="float64")
    return float((ranks[labels].sum() - positive_count * (positive_count + 1) / 2.0)
                 / (positive_count * negative_count))


def scan_threshold(
    values: Any,
    positive: Any,
    direction: str,
    quantiles: int = 200,
) -> Mapping[str, Any]:
    """扫描阈值并返回 F1 最优工作点。

    Args:
        values: 特征值数组。
        positive: 二值标签（1 = 目标类别）。
        direction: ``"le"`` 或 ``"ge"``。
        quantiles: 候选阈值个数（按数据分位点生成）。

    Returns:
        含 ``threshold`` / ``direction`` / ``precision`` / ``recall`` / ``f1`` / ``tp`` / ``fp`` / ``fn`` 的映射。

    Raises:
        ValueError: 方向非法。
    """
    import numpy as np

    if direction not in ("le", "ge"):
        raise ValueError(f"direction 需为 le/ge，实际为 {direction!r}")

    series = np.asarray(values, dtype="float64")
    labels = np.asarray(positive).astype(bool)
    candidates = np.unique(np.quantile(series, np.linspace(0.01, 0.99, max(2, quantiles))))

    best: dict[str, Any] | None = None
    for threshold in candidates:
        predicted = series <= threshold if direction == "le" else series >= threshold
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "direction": direction,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    assert best is not None  # candidates 非空
    best["auc"] = auc_score(series, labels, direction)
    return best


def calibrate_detector(
    frame: Any,
    detector_id: str,
    plan: Mapping[str, Any],
    quantiles: int = 200,
    top: int = 5,
) -> list[Mapping[str, Any]]:
    """对一个检测器的候选特征逐个标定。

    Args:
        frame: **验证集**特征表。
        detector_id: 检测器 id。
        plan: 标定计划（目标类别与候选特征）。
        quantiles: 阈值候选个数。
        top: 返回前 N 个（按 F1 降序）。

    Returns:
        标定结果列表，每项含检测器、特征、方向、阈值、P/R/F1、AUC。

    Raises:
        KeyError: 目标类别非法。
        ValueError: 标签列或全部候选特征缺失。
    """
    import numpy as np

    label_column = schema.LABEL_COLUMN
    if label_column not in frame.columns:
        raise ValueError(f"缺少标签列 {label_column!r}")

    target = str(plan["attack"])
    if target not in TARGET_LABELS:
        raise KeyError(f"未知目标类别 {target!r}")

    positive = (frame[label_column].to_numpy() == TARGET_LABELS[target])
    results: list[Mapping[str, Any]] = []

    for name in plan["features"]:
        if name not in frame.columns:
            continue
        series = frame[name].to_numpy(dtype="float64")
        # 变化率类特征在首行/缺口后会得到 NaN，必须按行剔除，
        # 否则分位点与判据都会被 NaN 污染（阈值会算成 nan）。
        finite = np.isfinite(series)
        usable = series[finite]
        usable_positive = positive[finite]
        if usable.size == 0 or float(usable.std()) == 0 or usable_positive.sum() == 0:
            continue
        for direction in ("le", "ge"):
            outcome = scan_threshold(usable, usable_positive, direction, quantiles)
            results.append(
                {
                    "detector": detector_id,
                    "feature": name,
                    "target": target,
                    **outcome,
                }
            )

    results.sort(key=lambda item: item["f1"], reverse=True)
    return results[:top]


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """把标定结果格式化为对齐的文本表。"""
    header = f"    {'特征':<22}{'AUC':>7}{'方向':>6}{'阈值':>12}{'P':>8}{'R':>8}{'F1':>8}"
    lines = [header, "    " + "-" * (len(header) - 4)]
    for row in rows:
        lines.append(
            f"    {row['feature']:<22}{row['auc']:>7.3f}{row['direction']:>6}"
            f"{row['threshold']:>12.4g}{row['precision']:>8.3f}{row['recall']:>8.3f}{row['f1']:>8.3f}"
        )
    return "\n".join(lines)


def emit_yaml(best_by_detector: Mapping[str, Mapping[str, Any]]) -> str:
    """生成可粘贴到 config.yaml 的 ``detectors`` 片段。

    Args:
        best_by_detector: 检测器 → 该检测器的最佳标定结果。

    Returns:
        YAML 文本。
    """
    lines = ["# ---- 由 scripts/calibrate_thresholds.py 在验证集上标定得到 ----",
             "# 说明：仅覆盖能从单特征标定的键；其余键（如 rules）需另行设置", "detectors:"]
    for detector_id, best in sorted(best_by_detector.items()):
        key = CALIBRATION_PLAN[detector_id]["config_keys"].get(best["feature"])
        if key is None:
            continue
        operator = "<=" if best["direction"] == "le" else ">="
        lines.append(f"  {detector_id}:")
        lines.append(
            f"    {key}: {best['threshold']:.6g}"
            f"    # 判据 {best['feature']} {operator} 阈值；"
            f"F1={best['f1']:.3f} AUC={best['auc']:.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    frame = load_features(args)

    # §16.2：按事件切分，保证验证集含所有类别
    split = split_by_events(
        frame, label_column=schema.LABEL_COLUMN, validation_event_ratio=args.validation_event_ratio
    )
    validation = frame.iloc[split.indices("validation")].reset_index(drop=True)
    print(f"[calibrate] 事件级切分：{split.summary()}")
    counts = validation[schema.LABEL_COLUMN].value_counts().sort_index()
    distribution = {schema.LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()}
    print(f"[calibrate] 验证集类别分布：{distribution}")
    if len(counts) < 2:
        print("[calibrate] 警告：验证集只有单一类别，标定结果不可用")
        return 1

    chosen = args.detector or sorted(CALIBRATION_PLAN)
    targets = [args.target] if args.target else None

    best_by_detector: dict[str, Mapping[str, Any]] = {}
    for detector_id in chosen:
        plan = dict(CALIBRATION_PLAN[detector_id])
        if targets is not None:
            if plan["attack"] not in targets:
                continue
        rows = calibrate_detector(validation, detector_id, plan, args.quantiles, args.top)
        if not rows:
            print(f"\n[{detector_id}] 没有可用特征（验证集缺少相应列或方差为 0）")
            continue
        print(f"\n[{detector_id}] 目标={plan['attack'].capitalize()}，"
              f"验证集正样本 {int((validation[schema.LABEL_COLUMN] == TARGET_LABELS[plan['attack']]).sum())} 行")
        print(format_table(rows))
        best = rows[0]
        best_by_detector[detector_id] = best
        verdict = "可用" if best["auc"] >= 0.7 else ("偏弱" if best["auc"] >= 0.6 else "几乎无区分度")
        print(f"    结论：最佳特征 {best['feature']}（AUC={best['auc']:.3f}，{verdict}）")

    print("\n[calibrate] 提示：")
    print("    - AUC < 0.6 的特征调阈值意义不大，应回到特征层（§6.3）；")
    print("    - 本结果只用验证集得到，请勿用留出/测试集标定（§16.3）；")
    print("    - 标定后请把阈值写入 config.yaml，并在 results/experiments_log.csv 留一行记录（§20.3）。")

    if args.emit_yaml:
        print("\n" + emit_yaml(best_by_detector))
    else:
        print("\n[calibrate] 加 --emit-yaml 可获得可直接粘贴到 config.yaml 的片段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
