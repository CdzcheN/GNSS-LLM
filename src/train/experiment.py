"""实验记录与配置哈希：让每次实验可定位、可复现、可对比。

对应开发文档
    §20.3 实验记录规范（experiment_id / config_hash / model_version / data_version /
    split_version / seed / metrics / figure_paths / checkpoint_path / conclusion）、
    §20.4 固定随机种子、§28 文档维护规则（变更需可追溯）。

职责
    1. 定义 ExperimentRecord，字段与 §20.3 完全一致；
    2. 由配置对象计算稳定的 ``config_hash``（规范化 JSON + SHA-256），作为复现锚点；
    3. 追加写入 ``results/experiments_log.csv``（字段顺序与仓库模板一致）。

不做（边界）
    - 不计算 metrics 本身（由 ``src/eval`` 负责）；
    - 不覆盖或改写既有记录（只追加，保证实验历史可回溯）；
    - 不判定实验结果好坏（结论由研究者填写）。

输入 / 输出
    输入：配置映射、指标映射、产物路径
    输出：一行 CSV 记录（以及可回读的字典）

关键约束
    - ``config_hash`` 必须对键序不敏感（排序后哈希），否则同一配置会产生不同哈希；
    - 非 JSON 原生对象（如 Path、dataclass）需可稳定序列化，使用 ``default=str`` 兜底；
    - 字段顺序由 ``EXPERIMENT_FIELDS`` 统一决定，禁止各实验自行调整列序。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: 实验记录字段（与 §20.3 及 results/experiments_log.csv 表头完全一致）。
EXPERIMENT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "config_hash",
    "model_version",
    "data_version",
    "split_version",
    "seed",
    "metrics",
    "figure_paths",
    "checkpoint_path",
    "conclusion",
)

#: 默认实验记录路径（与仓库内模板一致）。
DEFAULT_LOG_PATH: str = "results/experiments_log.csv"

#: 多值字段在 CSV 中的连接符（与事件日志保持一致）。
LIST_JOINER: str = "|"

#: config_hash 截断长度（便于人读，同时保留足够区分度）。
HASH_LENGTH: int = 12


def config_hash(config: Mapping[str, Any], length: int = HASH_LENGTH) -> str:
    """计算配置的稳定哈希。

    Args:
        config: 配置映射（可含嵌套结构）。
        length: 截断长度。

    Returns:
        十六进制哈希前缀。

    Raises:
        ValueError: ``length`` 不在 1..64 之间。
    """
    if not 1 <= length <= 64:
        raise ValueError(f"length 需在 1..64 之间，实际为 {length}")
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class ExperimentRecord:
    """一次实验的记录（字段与 §20.3 一致）。

    Attributes:
        experiment_id: 实验标识（同一配置重复运行应更换 id，见 §20.3）。
        config_hash: 配置哈希（由 ``config_hash`` 计算）。
        model_version: 模型版本（对应 §8.4 的 detector version）。
        data_version: 数据版本（数据来源与切分前的标识）。
        split_version: 划分版本（§16.2 的划分协议版本）。
        seed: 随机种子（§20.4）。
        metrics: 指标映射（§17 五层指标）。
        figure_paths: 图表路径序列。
        checkpoint_path: 权重路径（无则留空）。
        conclusion: 结论（由研究者填写，不得自动生成）。
    """

    experiment_id: str
    config_hash: str
    model_version: str = ""
    data_version: str = ""
    split_version: str = ""
    seed: int = 42
    metrics: Mapping[str, Any] = field(default_factory=dict)
    figure_paths: Sequence[str] = field(default_factory=tuple)
    checkpoint_path: str = ""
    conclusion: str = ""

    def __post_init__(self) -> None:
        """校验必填项，避免产生无法定位的实验行。"""
        if not self.experiment_id:
            raise ValueError("experiment_id 不能为空（§20.3）")
        if not self.config_hash:
            raise ValueError("config_hash 不能为空（§20.3 可复现锚点）")
        if not 0 <= int(self.seed):
            raise ValueError(f"seed 必须为非负整数：{self.seed}")

    def to_row(self) -> dict[str, Any]:
        """转换为 CSV 行（多值字段按 ``LIST_JOINER`` 连接，指标序列化为 JSON）。"""
        row: dict[str, Any] = {}
        for name in EXPERIMENT_FIELDS:
            value = getattr(self, name)
            if name == "metrics":
                row[name] = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
            elif isinstance(value, (list, tuple)):
                row[name] = LIST_JOINER.join(str(item) for item in value)
            else:
                row[name] = value
        return row


class ExperimentLog:
    """实验记录写入器（§20.3）。"""

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH) -> None:
        """初始化。

        Args:
            path: CSV 路径；目录不存在时自动创建。
        """
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> Mapping[str, Any]:
        """追加一条实验记录。

        Args:
            record: 实验记录。

        Returns:
            实际写入的行数据（便于调用方留痕）。

        Raises:
            OSError: 路径不可写。
        """
        row = record.to_row()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EXPERIMENT_FIELDS), extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        return row

    def read(self) -> Sequence[Mapping[str, str]]:
        """回读全部实验记录。

        Returns:
            行字典序列；文件不存在时返回空序列。
        """
        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))


def record_from_run(
    experiment_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    seed: int = 42,
    **extra: Any,
) -> ExperimentRecord:
    """便捷构造：由配置自动计算 config_hash。

    Args:
        experiment_id: 实验标识。
        config: 本次实验配置（用于计算哈希）。
        metrics: 指标映射。
        seed: 随机种子。
        **extra: 透传给 ``ExperimentRecord`` 的其它字段。

    Returns:
        ExperimentRecord。
    """
    return ExperimentRecord(
        experiment_id=experiment_id,
        config_hash=config_hash(config),
        seed=seed,
        metrics=dict(metrics or {}),
        **extra,
    )
