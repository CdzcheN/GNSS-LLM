"""测试 src/event：事件记录、状态机（§13.3）、告警合并（§13.2）与事件日志。

对应开发文档
    §13.1 事件字段、§13.2 合并窗口、§13.3 状态机、§15.1 结构化日志与告警、
    §14.4 C4 保留原始日志。

覆盖要点
    - 事件时间格式化与时长计算；
    - 状态迁移只允许 §13.3 的路径；
    - 合并窗口边界（gap == 窗口 → 仍合并；gap > 窗口 → 结束）；
    - 攻击类型变化必须切断事件，且“关旧开新”两次结果都要能拿到；
    - JSONL/CSV 落盘与回读，告警文本只来自结构化字段。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.detectors.base import AttackType
from src.event.event_logger import CSV_FIELDS, EventLogger, read_events
from src.event.event_manager import EventManager
from src.event.event_record import (
    ALLOWED_TRANSITIONS,
    EVENT_FIELDS,
    EventRecord,
    EventState,
    as_llm_input,
    format_seconds,
)
from src.llm.schema import validate_event


class EventRecordTest(unittest.TestCase):
    """§13.1 事件数据结构。"""

    def test_format_seconds(self) -> None:
        self.assertEqual(format_seconds(37295), "10:21:35")

    def test_duration_and_range_text(self) -> None:
        event = EventRecord("EV001", start_time=37295, end_time=37458)
        self.assertEqual(event.duration_s, 163.0)
        self.assertEqual(event.time_range_text(), "10:21:35 ~ 10:24:18")

    def test_duration_never_negative(self) -> None:
        self.assertEqual(EventRecord("EV001", start_time=100, end_time=50).duration_s, 0.0)

    def test_to_dict_contains_doc_fields(self) -> None:
        payload = EventRecord("EV001", start_time=0, end_time=1).to_dict()
        for name in EVENT_FIELDS:
            self.assertIn(name, payload)
        self.assertIn("state", payload)

    def test_state_transitions_follow_doc_133(self) -> None:
        event = EventRecord("EV001", start_time=0, end_time=0, state=EventState.NORMAL)
        self.assertTrue(event.can_transition_to(EventState.SUSPECTED))
        self.assertFalse(event.can_transition_to(EventState.ONGOING))
        self.assertIn(EventState.RECOVERED, ALLOWED_TRANSITIONS[EventState.ONGOING])

    def test_llm_input_passes_schema_contract(self) -> None:
        """事件 → LLM 输入的桥接结果必须能通过 §14.2 契约校验。"""
        event = EventRecord(
            "EV001", start_time=37295, end_time=37458,
            attack_type=AttackType.JAMMING, confidence=0.94,
            detection_delay_s=6, selected_strategies=("cno", "pvt"),
            key_evidence={"cn0_change": -8.7, "内部调试字段": 1},
        )
        payload = as_llm_input(event, data_quality=0.96)
        report = validate_event(payload)
        self.assertTrue(report.ok, report.messages)
        # 白名单外的证据必须被过滤，但仍保留在原始事件里（§14.4 C1/C4）
        self.assertIn("cn0_change", payload["key_features"])
        self.assertNotIn("内部调试字段", payload["key_features"])
        self.assertIn("内部调试字段", event.key_evidence)


class EventManagerTest(unittest.TestCase):
    """§13.2 合并与 §13.3 状态机。"""

    def test_normal_input_creates_no_event(self) -> None:
        manager = EventManager()
        update = manager.update(0.0, AttackType.NORMAL, 0.9)
        self.assertIsNone(update.current)
        self.assertIsNone(update.closed)
        self.assertFalse(update.opened_new)
        self.assertIsNone(manager.current)

    def test_first_detection_confirms_immediately(self) -> None:
        """默认 confirm_after_s=0：首次检出即 CONFIRMED（门限未由文档给出）。"""
        manager = EventManager()
        update = manager.update(100.0, AttackType.JAMMING, 0.9, ["cno"])
        self.assertTrue(update.opened_new)
        self.assertIs(update.current.state, EventState.CONFIRMED)
        self.assertEqual(update.current.event_id, "EV001")
        self.assertIn("cno", update.current.selected_strategies)

    def test_suspected_when_confirmation_delay_required(self) -> None:
        manager = EventManager(confirm_after_s=5.0)
        update = manager.update(100.0, AttackType.JAMMING, 0.9)
        self.assertIs(update.current.state, EventState.SUSPECTED)
        manager.update(106.0, AttackType.JAMMING, 0.9)
        self.assertIs(manager.current.state, EventState.CONFIRMED)

    def test_merge_within_window(self) -> None:
        manager = EventManager(merge_window_s=30.0)
        manager.update(1000.0, AttackType.JAMMING, 0.80, ["cno"])
        update = manager.update(1030.0, AttackType.JAMMING, 0.95, ["pvt"])
        event = update.current
        self.assertIsNone(update.closed)
        self.assertFalse(update.opened_new)
        self.assertEqual(event.samples, 2)
        self.assertEqual(event.duration_s, 30.0)
        self.assertEqual(event.confidence, 0.95)
        self.assertIn("pvt", event.selected_strategies)
        self.assertIs(event.state, EventState.ONGOING)

    def test_gap_beyond_window_closes_and_opens(self) -> None:
        manager = EventManager(merge_window_s=30.0)
        manager.update(0.0, AttackType.JAMMING, 0.9)
        update = manager.update(31.0, AttackType.JAMMING, 0.9)
        self.assertIs(update.closed.state, EventState.RECOVERED)
        self.assertEqual(update.closed.samples, 1)
        self.assertEqual(len(manager.finished), 1)
        self.assertTrue(update.opened_new)
        self.assertIsNotNone(update.current)
        self.assertIs(manager.current, update.current)

    def test_attack_type_change_splits_events(self) -> None:
        manager = EventManager()
        manager.update(0.0, AttackType.JAMMING, 0.9, ["cno"])
        update = manager.update(10.0, AttackType.SPOOFING, 0.8, ["observation"])
        self.assertIs(update.closed.state, EventState.RECOVERED)
        self.assertIs(update.closed.attack_type, AttackType.JAMMING)
        self.assertIs(update.current.attack_type, AttackType.SPOOFING)
        self.assertNotEqual(update.current.event_id, update.closed.event_id)

    def test_normal_after_attack_closes_event(self) -> None:
        manager = EventManager()
        manager.update(0.0, AttackType.JAMMING, 0.9)
        update = manager.update(5.0, AttackType.NORMAL, 0.99)
        self.assertIs(update.closed.state, EventState.RECOVERED)
        self.assertIsNone(update.current)
        self.assertIsNone(manager.current)

    def test_explicit_close(self) -> None:
        manager = EventManager()
        manager.update(0.0, AttackType.JAMMING, 0.9)
        closed = manager.close(100.0)
        self.assertIs(closed.state, EventState.RECOVERED)
        self.assertEqual(closed.end_time, 100.0)
        self.assertIsNone(manager.close())

    def test_illegal_confidence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventManager().update(0.0, AttackType.JAMMING, 1.5)

    def test_evidence_is_merged_across_windows(self) -> None:
        manager = EventManager()
        manager.update(0.0, AttackType.JAMMING, 0.9, ["cno"], {"cn0_change": -3.0})
        update = manager.update(10.0, AttackType.JAMMING, 0.9, ["pvt"], {"pdop_change": 2.4})
        self.assertEqual(
            dict(update.current.key_evidence),
            {"cn0_change": -3.0, "pdop_change": 2.4},
        )

    def test_event_ids_are_deterministic(self) -> None:
        manager = EventManager()
        manager.update(0.0, AttackType.JAMMING, 0.9)
        manager.update(100.0, AttackType.SPOOFING, 0.9)
        self.assertEqual([event.event_id for event in manager.finished], ["EV001"])


class EventLoggerTest(unittest.TestCase):
    """§15.1 结构化日志与告警（§14.4 C4）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_jsonl_and_csv_written(self) -> None:
        logger = EventLogger(self.tmp / "events.jsonl", self.tmp / "events.csv")
        event = EventRecord(
            "EV001", start_time=37295, end_time=37458,
            attack_type=AttackType.SPOOFING, confidence=0.94,
            detection_delay_s=6, selected_strategies=("observation", "pvt"),
            key_evidence={"cn0_change": -8.7}, state=EventState.RECOVERED,
        )
        logger.log(event)

        lines = (self.tmp / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        for name in CSV_FIELDS:
            self.assertIn(name, payload)

        header = (self.tmp / "events.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), list(CSV_FIELDS))

    def test_append_does_not_overwrite(self) -> None:
        logger = EventLogger(self.tmp / "events.jsonl", None)
        logger.log(EventRecord("EV001", start_time=0, end_time=1))
        logger.log(EventRecord("EV002", start_time=2, end_time=3))
        self.assertEqual(len(read_events(self.tmp / "events.jsonl")), 2)

    def test_read_events_missing_file_returns_empty(self) -> None:
        self.assertEqual(read_events(self.tmp / "none.jsonl"), ())

    def test_alert_text_comes_from_structured_fields(self) -> None:
        event = EventRecord(
            "EV001", start_time=37295, end_time=37458,
            attack_type=AttackType.SPOOFING, confidence=0.94,
            selected_strategies=("observation",), state=EventState.ONGOING,
        )
        text = EventLogger.alert_text(event)
        self.assertIn("EV001", text)
        self.assertIn("SPOOFING", text)
        self.assertIn("10:21:35 ~ 10:24:18", text)
        self.assertIn("conf=0.94", text)


if __name__ == "__main__":
    unittest.main()
