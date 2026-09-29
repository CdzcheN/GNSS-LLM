"""测试 src/llm：§14.2 输入契约、§14.3 报告模板与 §14.4 约束。

对应开发文档
    §14.2 LLM 输入、§14.3 LLM 输出、§14.4 约束 C1–C5、§18.8 LLM 模块验证。

覆盖要点
    - §14.2 的示例事件必须通过校验；
    - 白名单外字段（如原始数据）必须被拒绝，守住 §14.4 C1；
    - 报告模板首行与字段一致；摘要记录必须带可复现元数据（§14.4 C5）。
"""

from __future__ import annotations

import unittest

from src.llm.prompt import PROMPT_VERSION, build_prompt
from src.llm.report import MISSING, build_report
from src.llm.schema import validate_event
from src.llm.summarizer import TemplateSummarizer

#: §14.2 文档给出的输入示例（原样使用，作为契约基准）。
DOC_EVENT = {
    "event_id": "20260927_001",
    "start_time": "10:21:35",
    "end_time": "10:24:18",
    "duration_s": 163,
    "state": "spoofing",
    "confidence": 0.94,
    "selected_strategies": ["observation_consistency", "pvt_detection", "deep_temporal"],
    "detection_delay_s": 6,
    "key_features": {
        "cn0_change": -8.7,
        "satellite_change": -3,
        "pdop_change": 2.4,
        "position_error": 18.6,
    },
    "data_quality": 0.96,
}


class SchemaTest(unittest.TestCase):
    """§14.2 输入契约。"""

    def test_doc_example_passes(self) -> None:
        report = validate_event(DOC_EVENT)
        self.assertTrue(report.ok, report.messages)

    def test_out_of_whitelist_field_reported(self) -> None:
        report = validate_event({**DOC_EVENT, "raw_iq": [0.1, 0.2]})
        self.assertFalse(report.ok)
        self.assertIn("raw_iq", report.unknown_fields)

    def test_missing_required_field_reported(self) -> None:
        report = validate_event({"event_id": "x"})
        self.assertFalse(report.ok)
        self.assertIn("start_time", report.missing_fields)

    def test_unknown_key_feature_reported(self) -> None:
        report = validate_event({**DOC_EVENT, "key_features": {"unknown_metric": 1}})
        self.assertFalse(report.ok)
        self.assertIn("unknown_metric", report.unknown_key_features)

    def test_invalid_state_reported(self) -> None:
        report = validate_event({**DOC_EVENT, "state": "unknown_state"})
        self.assertFalse(report.ok)
        self.assertEqual(report.invalid_state, "unknown_state")


class ReportTest(unittest.TestCase):
    """§14.3 输出 3：监测报告格式。"""

    def test_report_starts_with_event_id_line(self) -> None:
        self.assertTrue(build_report(DOC_EVENT).startswith("事件编号：20260927_001"))

    def test_report_contains_time_range(self) -> None:
        report = build_report(DOC_EVENT)
        self.assertIn("10:21:35 ~ 10:24:18", report)

    def test_missing_fields_marked_explicitly(self) -> None:
        report = build_report({"event_id": "x", "state": "jamming"})
        self.assertIn(MISSING, report)


class SummarizerTest(unittest.TestCase):
    """摘要器行为（§14.4 C4/C5、§15.2 降级路径）。"""

    def test_template_summarizer_records_provenance(self) -> None:
        record = TemplateSummarizer().summarize(DOC_EVENT)
        self.assertEqual(record.input_event_id, "20260927_001")
        self.assertEqual(record.prompt_version, PROMPT_VERSION)
        self.assertTrue(record.report_text)

    def test_template_summarizer_rejects_invalid_event(self) -> None:
        with self.assertRaises(ValueError):
            TemplateSummarizer().summarize({"event_id": "x", "state": "spoofing", "confidence": 0.9, "raw_iq": []})

    def test_record_is_serializable(self) -> None:
        payload = TemplateSummarizer().summarize(DOC_EVENT).to_dict()
        for key in ("input_event_id", "summary_text", "model_name", "model_version", "prompt_version", "timestamp"):
            self.assertIn(key, payload)


class PromptTest(unittest.TestCase):
    """提示词约束与版本可追溯（§14.4、§2.4）。"""

    def test_prompt_carries_version_and_constraints(self) -> None:
        prompt = build_prompt(DOC_EVENT)
        self.assertIn(PROMPT_VERSION, prompt)
        self.assertIn("不得修改事件时间", prompt)
        self.assertIn("20260927_001", prompt)

    def test_prompt_requires_event_id(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt({})

    def test_prompt_rejects_empty_event(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt({})


if __name__ == "__main__":
    unittest.main()
