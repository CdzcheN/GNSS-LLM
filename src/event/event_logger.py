"""M7 事件日志与告警输出：结构化事件的持久化与实时告警文本。

对应开发文档
    §13.1 事件字段、§15.1 在线流程（结构化日志 → 实时告警）、
    §14.4 C4 保留原始日志、§19.4 LLM 不进入关键路径、§20.1 日志格式（CSV / JSON）。

职责
    1. 把事件以 JSON Lines 追加落盘（原始日志，供复现与审计，§14.4 C4）；
    2. 可选输出 CSV 汇总（字段顺序与 §13.1 一致，便于表格分析）；
    3. 生成不依赖 LLM 的实时告警文本（§15.2：LLM 不可用也能告警）。

不做（边界）
    - 不生成自然语言摘要（属 M8，LLM 或模板摘要器）；
    - 不修改事件内容（只做序列化）；
    - 不阻塞实时链路（写入方式为短小的追加写，失败由调用方决定是否降级）。

输入 / 输出
    输入：EventRecord
    输出：JSONL / CSV 文件、告警文本

关键约束
    - 日志字段必须完整保留 §13.1 的全部字段（缺失即视为日志缺陷）；
    - 追加写不得覆盖历史记录，保证实验可回溯（§20.3）；
    - 告警文本必须来自结构化字段，不得包含推测性描述（§2.4）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.event.event_record import EventRecord

#: CSV 列顺序（§13.1 字段 + 状态）。
CSV_FIELDS: tuple[str, ...] = (
    "event_id",
    "start_time",
    "end_time",
    "duration_s",
    "attack_type",
    "confidence",
    "detection_delay_s",
    "selected_strategies",
    "key_evidence",
    "state",
)

#: 多值字段在 CSV 中的连接符。
LIST_JOINER: str = "|"

#: 默认输出路径。
DEFAULT_JSONL_PATH: str = "results/events.jsonl"
DEFAULT_CSV_PATH: str = "results/events.csv"


class EventLogger:
    """事件日志写入器（§14.4 C4、§20.1）。"""

    def __init__(
        self,
        jsonl_path: str | Path = DEFAULT_JSONL_PATH,
        csv_path: str | Path | None = DEFAULT_CSV_PATH,
    ) -> None:
        """初始化。

        Args:
            jsonl_path: JSON Lines 输出路径（目录不存在时自动创建）。
            csv_path: 可选的 CSV 汇总路径；``None`` 表示只写 JSONL。
        """
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path) if csv_path is not None else None

    def log(self, event: EventRecord) -> Mapping[str, Any]:
        """把事件追加写入 JSONL 与（可选）CSV。

        Args:
            event: 事件记录。

        Returns:
            实际写入的结构化字典（便于调用方留痕）。

        Raises:
            OSError: 路径不可写。
        """
        payload = event.to_dict()

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                writer.writerow(self._to_csv_row(payload))

        return payload

    def log_alert(self, event: EventRecord) -> str:
        """输出实时告警文本并返回该文本（§15.1、§15.2）。

        Args:
            event: 事件记录。

        Returns:
            告警文本（只含结构化字段，不含推测）。
        """
        text = self.alert_text(event)
        print(text)
        return text

    @staticmethod
    def alert_text(event: EventRecord) -> str:
        """生成结构化告警文本。

        Args:
            event: 事件记录。

        Returns:
            形如 ``[ALERT] EV001 JAMMING 10:21:35 ~ 10:24:18 (163s) conf=0.94`` 的文本。
        """
        return (
            f"[ALERT] {event.state.value.upper()} {event.event_id} "
            f"{event.attack_type.name} {event.time_range_text()} "
            f"({event.duration_s:.0f}s) conf={event.confidence:.2f} "
            f"strategies={list(event.selected_strategies)}"
        )

    @staticmethod
    def _to_csv_row(payload: Mapping[str, Any]) -> dict[str, Any]:
        """把事件字典转换为 CSV 行（多值字段用 ``|`` 连接）。"""
        row: dict[str, Any] = {}
        for name in CSV_FIELDS:
            value = payload.get(name)
            if isinstance(value, (list, tuple)):
                row[name] = LIST_JOINER.join(str(item) for item in value)
            elif isinstance(value, Mapping):
                row[name] = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
            else:
                row[name] = value
        return row


def read_events(jsonl_path: str | Path = DEFAULT_JSONL_PATH) -> Sequence[Mapping[str, Any]]:
    """读取 JSONL 事件日志（用于分析与指标计算，§17.3）。

    Args:
        jsonl_path: JSONL 路径。

    Returns:
        事件字典序列；文件不存在时返回空序列。
    """
    path = Path(jsonl_path)
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())
