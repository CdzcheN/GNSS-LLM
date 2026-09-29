"""脚本：多源数据解析与特征提取，产出统一特征表（M1 → M2）。

对应开发文档
    §6.1 标准处理流水线、§5.3 模态定义、§6.2 时间窗口、§6.3 派生特征、§6.4 标准化。

用法
    # 数据集概况（不写文件）
    python -m scripts.extract_features --describe

    # 三态数据小样本（快速验证链路）
    python -m scripts.extract_features --dataset three_state --nrows 5000 \
        --out data/features_1221_head.csv

    # 单个正常数据文件
    python -m scripts.extract_features --dataset normal --file gnss_complete_featuresObsSatPvt_12-16.csv \
        --out data/features_12-16.csv

流程（§6.1 顺序）
    1. 解析 CSV 并校验 schema           → src/data/parser.py
    2. 时间排序与重复历元清理            → src/data/alignment.py
    3. 质量掩码（sat_mask / miss_mask） → src/data/quality.py
    4. 特征提取（信号 / 导航 / 观测）    → src/features/*
    5. 写出特征表（CSV 或 Parquet）

尚未实现（数据未提供，运行时会给出明确提示）
    - 模态 B（MON-SPAN 频谱 / AGC）与模态 C 的多普勒、载波相位特征；
    - 标准化 scaler 的拟合与保存（§6.4：统计量只能来自训练集，不应在本脚本对全量数据拟合）。

关键约束
    - 输出必须保留时间列与标签列，便于按 §16.2 划分；
    - 不做随机打散（§2.5、§16.3）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # 支持 `python scripts/extract_features.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import alignment, parser, quality, schema  # noqa: E402

#: 数据集选项。
DATASET_CHOICES: tuple[str, ...] = ("normal", "three_state")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="多源 GNSS 特征提取（开发文档 §6.1）")
    ap.add_argument("--dataset", default="three_state", choices=DATASET_CHOICES,
                    help="数据集：normal（9 月 12–30 日）或 three_state（1221）")
    ap.add_argument("--file", help="仅处理指定文件（normal 数据集；默认处理全部）")
    ap.add_argument("--nrows", type=int, default=None, help="每个文件读取的行数上限")
    ap.add_argument("--out", default=None, help="输出路径（.csv 或 .parquet）")
    ap.add_argument("--describe", action="store_true", help="只打印数据集概况")
    ap.add_argument("--residual-threshold", type=float, default=15.0,
                    help="伪距残差超限阈值（仅用于生成 res_outlier_count 特征）")
    return ap


def extract_features(name: str, nrows: int | None, residual_threshold: float) -> Any:
    """执行 §6.1 的流水线并返回特征表。

    Args:
        name: 数据集文件名。
        nrows: 读取行数上限。
        residual_threshold: 残差超限阈值（来自命令行，未标定）。

    Returns:
        含原始列、掩码列与派生特征的 DataFrame。

    Raises:
        ImportError: 未安装 pandas。
        ValueError: 列名与 schema 不一致。
    """
    pd = _require_pandas()
    from src.features import navigation, observation, signal

    raw = parser.load_features_csv(name, nrows=nrows)
    ordered = alignment.sort_by_time(raw)
    deduplicated = alignment.drop_duplicate_timestamps(ordered)
    masked = quality.attach_masks(deduplicated)

    features = pd.concat(
        [
            masked,
            signal.signal_features(masked),
            navigation.navigation_features(masked),
            observation.observation_features(masked, outlier_threshold=residual_threshold),
        ],
        axis=1,
    )
    return features


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def _write(frame: Any, path: str) -> None:
    """按扩展名写出特征表。

    Args:
        frame: 特征表。
        path: 输出路径。

    Raises:
        ValueError: 不支持的扩展名。
        ImportError: 写出 Parquet 但未安装 pyarrow。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(target, index=False)
    elif suffix in (".parquet", ".pq"):
        try:
            frame.to_parquet(target, index=False)
        except ImportError as exc:  # pragma: no cover - 依赖环境相关
            raise ImportError(
                "写出 Parquet 需要 pyarrow（requirements.txt 已列）；"
                "也可改用 --out xxx.csv"
            ) from exc
    else:
        raise ValueError(f"不支持的输出格式 {suffix!r}，请使用 .csv 或 .parquet")
    print(f"[extract_features] 已写出 {target}（{len(frame)} 行 × {frame.shape[1]} 列）")


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    if args.describe:
        print("[extract_features] 数据集概况：")
        for name, info in parser.describe_dataset().items():
            status = "存在" if info["exists"] else "缺失"
            print(f"  {status}  {name}  {info['size_mb']} MB  期望行数 {info['expected_rows']}")
        print(f"[extract_features] schema：{schema.EXPECTED_COLUMN_COUNT} 列 / "
              f"{schema.EXPECTED_FEATURE_COUNT} 个数值特征")
        return 0

    if args.dataset == "normal":
        targets = [args.file] if args.file else list(schema.NORMAL_FILES)
        outputs = []
        for name in targets:
            features = extract_features(name, args.nrows, args.residual_threshold)
            print(f"[extract_features] {name}：{features.shape[0]} 行 × {features.shape[1]} 列")
            outputs.append((name, features))
        if args.out:
            if len(outputs) == 1:
                _write(outputs[0][1], args.out)
            else:
                print("[extract_features] 多个输入文件时请配合 --file 指定单个，或分次运行")
        return 0

    features = extract_features(schema.THREE_STATE_FILE, args.nrows, args.residual_threshold)
    print(f"[extract_features] {schema.THREE_STATE_FILE}：{features.shape[0]} 行 × {features.shape[1]} 列")
    print(f"[extract_features] 标签分布：{features[schema.LABEL_COLUMN].value_counts().to_dict()}")
    if args.out:
        _write(features, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
