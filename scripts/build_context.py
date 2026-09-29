"""脚本：由特征表构建 GNSS Context 序列（M2 → M3）。

对应开发文档
    §7 章（7.2 Context 组成、7.3 上下文特征、7.4 上下文输出）、
    §18.1 Context 消融实验、§9.2 决策输入。

用法
    # 由特征表生成 Context 列（展平后附加到原表，便于与标签对齐）
    python -m scripts.build_context --features data/features_1221_head.csv \
        --out data/context_1221_head.csv

    # §18.1 的“无 Context”对照组：只复制原表并标注 enabled=False
    python -m scripts.build_context --features data/features_1221_head.csv --no-context --out data/nocontext.csv

流程
    1. 读取 M2 特征表（须含掩码与派生特征列）
    2. 按列语义分派到六分量 S/Q/O/N/H/D（§7.2）
    3. 估计上下文置信度并透传数据质量（§7.4）
    4. 展平为 ``S.cn0_delta_db`` 形式的列并写出（可选保留原列）

关键约束
    - 历史分量只使用当前行之前的窗口（§16.3 禁止未来信息）；
    - ``D``（检测器历史）需由运行期注入，本脚本产出时为空——这是预期行为；
    - 输出保留时间列与标签列，便于按 §16.2 划分与训练。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # 支持 `python scripts/build_context.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context.context_encoder import (  # noqa: E402
    CONTEXT_COMPONENTS,
    ContextEncoder,
    describe_components,
)
from src.data import quality, schema  # noqa: E402

#: 默认保留的原表列（时间、标签与身份列）。
KEEP_COLUMNS: tuple[str, ...] = (
    schema.TIME_COLUMN,
    schema.DAY_COLUMN,
    schema.HOUR_COLUMN,
    schema.LABEL_COLUMN,
)

#: 默认历史窗口（历元数）。
DEFAULT_HISTORY_WINDOWS: int = 5


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="构建 GNSS Context 序列（开发文档 §7）")
    ap.add_argument("--features", required=True, help="M2 特征表路径（CSV）")
    ap.add_argument("--out", default="data/context.csv", help="输出 CSV 路径")
    ap.add_argument("--history-windows", type=int, default=DEFAULT_HISTORY_WINDOWS,
                    help="历史分量 H_t 保留的历元数")
    ap.add_argument("--nrows", type=int, default=None, help="读取行数上限")
    ap.add_argument("--step", type=int, default=1, help="编码采样步长（>1 时按行抽样）")
    ap.add_argument("--no-context", dest="enabled", action="store_false",
                    help="禁用 Context 编码（§18.1 对照组）")
    ap.add_argument("--print-components", action="store_true", help="只打印六分量说明")
    ap.set_defaults(enabled=True)
    return ap


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def build_context_frame(frame: Any, encoder: ContextEncoder, step: int = 1) -> Any:
    """把特征表编码为 Context 并展平为列。

    Args:
        frame: M2 特征表（含掩码，若有 ``data_quality`` 列会作为上下文质量使用）。
        encoder: ContextEncoder 实例。
        step: 采样步长。

    Returns:
        含 ``context_confidence`` 与各分量展平列的 DataFrame（索引与采样行对齐）。

    Raises:
        ValueError: ``step`` 非正。
    """
    pd = _require_pandas()
    if step < 1:
        raise ValueError(f"step 必须 >= 1：{step}")

    rows: list[dict[str, Any]] = []
    indexes: list[Any] = []
    for index in range(0, len(frame), step):
        context = encoder.encode_from_frame(frame, index)
        flattened = context.flattened() if context is not None else {}
        record: dict[str, Any] = {
            "context_enabled": encoder.enabled,
            "context_confidence": None if context is None else context.confidence,
            "context_data_quality": None if context is None else context.data_quality,
        }
        record.update(flattened)
        rows.append(record)
        indexes.append(frame.index[index])

    return pd.DataFrame(rows, index=indexes)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    if args.print_components:
        print(f"[build_context] 六分量：{CONTEXT_COMPONENTS}")
        for name, meaning in describe_components().items():
            print(f"  {name}: {meaning}")
        return 0

    pd = _require_pandas()
    source = Path(args.features)
    if not source.exists():
        print(f"[build_context] 特征表不存在：{source}（请先运行 scripts/extract_features.py）")
        return 1

    frame = pd.read_csv(source, nrows=args.nrows)
    if schema.TIME_COLUMN in frame.columns:
        frame = quality.attach_masks(frame)

    encoder = ContextEncoder(enabled=args.enabled, history_windows=args.history_windows)
    contexts = build_context_frame(frame, encoder, step=args.step)

    keep = [name for name in KEEP_COLUMNS if name in frame.columns]
    output = pd.concat([frame.loc[contexts.index, keep], contexts], axis=1)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(f"[build_context] 已写出 {target}（{len(output)} 行 × {output.shape[1]} 列）")
    print(f"[build_context] Context 启用：{encoder.enabled}；"
          f"分量列数：{ {name: sum(1 for c in output.columns if c.startswith(f'{name}.')) for name in CONTEXT_COMPONENTS} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
