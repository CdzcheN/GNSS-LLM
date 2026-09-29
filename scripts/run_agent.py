"""脚本：运行 Agent 闭环 —— 感知 → 决策 → 检测 → 融合 → 事件 → 上报（§9、§12–§15）。

对应开发文档
    §9 智能体动态策略选择、§12 多策略结果融合、§13 异常事件管理、
    §15 实时监测流程、§19.3 Agent 部署原则（轻量、有限工具集、有限决策步数）。

用法
    # 小样本跑通闭环并产出预测文件（供 scripts/evaluate.py 评估）
    python -m scripts.run_agent --features data/features_1221_head.csv \
        --nrows 5000 --policy rule --out-predictions results/pred_1221_head.csv

    # §18.2 的“固定策略”对照组
    python -m scripts.run_agent --features data/features_1221_head.csv --policy fixed

    # 只装配组件并自检（不读数据）
    python -m scripts.run_agent --dry-run

闭环步骤（§15.1）
    1. 读取特征表（含掩码与派生特征）
    2. 逐历元构建 Context（M3）
    3. 策略选择得到 Plan（M5）
    4. 执行检测器，支持置信度提前终止（M4 + §9.4）
    5. 结果融合并识别冲突（M6）
    6. 更新事件状态、写结构化日志与告警（M7）
    7. 汇总统计并（可选）写出预测文件

关键约束
    - 特征表必须来自 ``scripts/extract_features.py``（已含 ``sat_mask`` 等掩码列）；
    - 检测器阈值取自 ``config.yaml`` 的 ``detectors`` 段，本脚本不硬编码判据；
    - 事件状态只由结构化检测逻辑产生，LLM 不参与（§13.3、§2.4）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):  # 支持 `python scripts/run_agent.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.executor import Executor  # noqa: E402
from src.agent.planner import FALLBACK_DETECTOR  # noqa: E402
from src.agent.policy import POLICY_LEVELS, ContextRulePolicy, FixedPolicy  # noqa: E402
from src.agent.tool_registry import DetectorRegistry  # noqa: E402
from src.context.context_encoder import ContextEncoder, FlagThresholds  # noqa: E402
from src.data import schema  # noqa: E402
from src.detectors.cno import CnoDetector  # noqa: E402
from src.detectors.deep_temporal import DeepTemporalDetector  # noqa: E402
from src.detectors.observation import ObservationDetector  # noqa: E402
from src.detectors.pvt import PvtDetector  # noqa: E402
from src.detectors.satellite import SatelliteDetector  # noqa: E402
from src.detectors.spectrum import SpectrumDetector  # noqa: E402
from src.detectors.threshold import ThresholdDetector  # noqa: E402
from src.event.event_logger import EventLogger  # noqa: E402
from src.event.event_manager import EventManager  # noqa: E402
from src.fusion.result_fusion import fuse  # noqa: E402
from src.llm.summarizer import TemplateSummarizer  # noqa: E402

#: 策略选项（§9.6 的 Level 0 与 Level 1）。
POLICY_CHOICES: tuple[str, ...] = ("fixed", "rule")

#: 逐历元预测结果中一并带出的关键特征（用于出图诊断“特征是否可分”，§6.3、§17）。
DEFAULT_PREDICTION_FEATURES: tuple[str, ...] = (
    "cn0_valid_mean",
    "cn0_delta_db",
    "cn0_valid_count",
    "res_valid_mean",
    "res_outlier_count",
    "pDOP_vs_baseline",
    "hAcc_vs_baseline",
    "valid_sat_count",
)


def build_registry(checkpoint: str | None = None) -> DetectorRegistry:
    """按 §8.2 注册全部检测器（S1–S7）。

    Args:
        checkpoint: 深度检测器的权重路径（可选）。

    Returns:
        已注册 S1–S7 的注册表（§8.4）。
    """
    registry = DetectorRegistry()
    for detector in (
        ThresholdDetector(),                                  # S1
        SpectrumDetector(),                                   # S2
        CnoDetector(),                                        # S3
        SatelliteDetector(),                                  # S4
        ObservationDetector(),                                # S5
        PvtDetector(),                                        # S6
        DeepTemporalDetector(checkpoint=checkpoint),          # S7
    ):
        registry.register(detector)
    return registry


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="运行 Agent 闭环（开发文档 §9、§12–§15）")
    ap.add_argument("--config", default="config.yaml", help="全局配置（§20.1 YAML）")
    ap.add_argument("--features", default="data/features_1221_head.csv",
                    help="特征表（由 scripts/extract_features.py 生成）")
    ap.add_argument("--nrows", type=int, default=None, help="读取行数上限")
    ap.add_argument("--policy", default="rule", choices=POLICY_CHOICES,
                    help="策略层级：fixed=Level 0 固定策略，rule=Level 1 上下文规则路由（§9.6）")
    ap.add_argument("--step", type=int, default=1, help="逐历元推进的步长（>1 表示抽样）")
    ap.add_argument("--stop-confidence", type=float, default=None,
                    help="达到该置信度即停止追加检测（§9.4 终止分支）")
    ap.add_argument("--checkpoint", default=None, help="深度检测器权重路径（可选）")
    ap.add_argument("--events-out", default="results/events.jsonl", help="事件日志输出（§14.4 C4）")
    ap.add_argument("--events-csv", default="results/events.csv", help="事件 CSV 输出（可为空字符串）")
    ap.add_argument("--out-predictions", default=None, help="逐历元预测输出 CSV（供 evaluate 使用）")
    ap.add_argument("--no-context", dest="use_context", action="store_false",
                    help="禁用 Context（§18.1 对照组）")
    ap.add_argument("--dry-run", action="store_true", help="只装配组件并打印计划，不读数据")
    ap.set_defaults(use_context=True)
    return ap


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def load_config(path: str) -> Mapping[str, Any]:
    """读取 YAML 配置。

    Args:
        path: 配置文件路径。

    Returns:
        配置映射。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError: 未安装 PyYAML。
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 PyYAML，请执行 `pip install -r requirements.txt`") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_closed_loop(
    frame: Any,
    registry: DetectorRegistry,
    policy: Any,
    encoder: ContextEncoder,
    manager: EventManager,
    logger: EventLogger,
    detector_configs: Mapping[str, Any],
    step: int = 1,
    stop_confidence: float | None = None,
    feature_columns: Sequence[str] = DEFAULT_PREDICTION_FEATURES,
) -> Mapping[str, Any]:
    """逐历元执行闭环（§15.1）。

    Args:
        frame: 特征表（含派生特征列）。
        registry: 检测器注册表。
        policy: 策略实例（Level 0 或 Level 1）。
        encoder: Context 编码器。
        manager: 事件管理器。
        logger: 事件日志写入器。
        detector_configs: ``detectors`` 段配置（各检测器阈值）。
        step: 采样步长。
        stop_confidence: 置信度提前终止门限。
        feature_columns: 逐历元结果中一并带出的特征列（供出图与误判分析）。

    Returns:
        统计映射（历元数、事件数、检测器调用统计、预测分布）。

    Raises:
        ValueError: ``step`` 非正。
    """
    if step < 1:
        raise ValueError(f"step 必须 >= 1：{step}")

    executor = Executor(registry)
    predictions: list[dict[str, Any]] = []
    call_count = 0
    conflict_count = 0
    events = 0

    for index in range(0, len(frame), step):
        row = frame.iloc[index]
        data = {
            key: (None if _is_na(value) else value)
            for key, value in row.to_dict().items()
        }
        context = encoder.encode_from_frame(frame, index)

        plan = policy.select(context=context, available=registry.ids())
        results = executor.execute(
            plan,
            data=data,
            context=context,
            configs=detector_configs,
            stop_on_confidence=stop_confidence,
        )
        call_count += len(results)

        fused = fuse(results)
        if fused.conflict:
            conflict_count += 1

        update = manager.update(
            float(index),  # 1 Hz 数据下，行号即秒数
            fused.attack_type,
            fused.confidence,
            list(fused.used_detectors),
            key_evidence={"contributions": dict(fused.contributions)},
        )
        if update.closed is not None:
            events += 1
            logger.log(update.closed)
            logger.log_alert(update.closed)

        record = {
            schema.LABEL_COLUMN: data.get(schema.LABEL_COLUMN),
            "prediction": int(fused.attack_type),
            "confidence": float(fused.confidence),
            "conflict": bool(fused.conflict),
            "detectors": ",".join(fused.used_detectors),
            "timestamp": str(data.get(schema.TIME_COLUMN)),
        }
        # 一并带出关键特征值：出图时可直接观察“异常区间内特征是否真的变化”（§6.3）
        for name in feature_columns:
            value = data.get(name)
            record[name] = None if value is None or _is_na(value) else float(value)
        predictions.append(record)

    closed = manager.close(float(len(frame)))
    if closed is not None:
        events += 1
        logger.log(closed)
        logger.log_alert(closed)

    return {
        "epochs": len(predictions),
        "events": events,
        "avg_detector_calls": call_count / len(predictions) if predictions else 0.0,
        "conflict_rate": conflict_count / len(predictions) if predictions else 0.0,
        "predictions": predictions,
    }


def _is_na(value: Any) -> bool:
    """判断值是否为 NaN/NaT（避免写 JSON 时出现非法数值）。"""
    try:
        return value != value  # NaN 与 NaT 的常见判定
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    registry = build_registry(args.checkpoint)
    policy = FixedPolicy() if args.policy == "fixed" else ContextRulePolicy()
    encoder = ContextEncoder(
        enabled=args.use_context,
        history_windows=5,
        thresholds=FlagThresholds.from_config(config.get("detectors")),
    )

    print(f"[run_agent] 已注册检测器：{registry.ids()}")
    print(f"[run_agent] 策略：{policy.describe()}（层级 {POLICY_LEVELS.get(policy.level)}）")
    print(f"[run_agent] Context 启用：{encoder.enabled}")

    if args.dry_run:
        plan = policy.select(available=registry.ids())
        print(f"[run_agent] 计划：{plan.selected_detectors} | 理由：{plan.selection_reason}")
        results = Executor(registry).execute(plan, data={"cn0_delta_db": -8.0}, configs={})
        fused = fuse(results)
        print(f"[run_agent] 融合：{fused.attack_type.name} conf={fused.confidence:.3f} "
              f"conflict={fused.conflict}")
        summary = TemplateSummarizer().summarize(
            {
                "event_id": "dry_run",
                "start_time": "00:00:00",
                "end_time": "00:00:01",
                "state": "jamming",
                "confidence": 0.9,
                "selected_strategies": list(plan.selected_detectors),
            }
        ).summary_text
        print(f"[run_agent] 摘要（降级模板）：{summary}")
        return 0

    features_path = Path(args.features)
    if not features_path.exists():
        print(f"[run_agent] 特征表不存在：{features_path}")
        print("[run_agent] 请先运行：python -m scripts.extract_features --dataset three_state "
              f"--nrows 5000 --out {args.features}")
        return 1

    pd = _require_pandas()
    frame = pd.read_csv(features_path, nrows=args.nrows)
    print(f"[run_agent] 特征表：{features_path}（{len(frame)} 行 × {frame.shape[1]} 列）")

    manager = EventManager(
        merge_window_s=float((config.get("event") or {}).get("alert_merge_window_s", 30.0)),
        confirm_after_s=float((config.get("event") or {}).get("confirm_after_s", 0.0)),
    )
    logger = EventLogger(args.events_out, args.events_csv or None)
    detector_configs = dict((config.get("detectors") or {}))

    stats = run_closed_loop(
        frame=frame,
        registry=registry,
        policy=policy,
        encoder=encoder,
        manager=manager,
        logger=logger,
        detector_configs=detector_configs,
        step=args.step,
        stop_confidence=args.stop_confidence,
    )

    print(f"[run_agent] 处理历元：{stats['epochs']}，形成事件：{stats['events']}")
    print(f"[run_agent] 平均检测器调用数：{stats['avg_detector_calls']:.2f}，"
          f"冲突率：{stats['conflict_rate']:.4f}")
    print(f"[run_agent] 事件日志：{args.events_out}")

    if args.out_predictions:
        target = Path(args.out_predictions)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats["predictions"]).to_csv(target, index=False)
        print(f"[run_agent] 预测已写出：{target}"
              f"（可用 scripts/evaluate.py --predictions {target} 评估）")
        print("[run_agent] 示例命令：python -m scripts.evaluate --predictions "
              f"{target} --pred-column prediction")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
