"""测试 src/context 与 src/data/quality：Context 六分量与基础数据质量判据。

对应开发文档
    §7.2 Context 组成、§7.4 上下文输出、§18.1 Context 消融、§5.4 Q3 卫星掩码。

说明
    quality.sat_mask 接受任意支持 ``>`` 比较的对象（设计上不依赖 pandas），
    因此这里用最小替身对象验证判据，无需安装 pandas 即可回归。
"""

from __future__ import annotations

import unittest
from typing import Sequence

from src.context.context_encoder import CONTEXT_COMPONENTS, Context, ContextEncoder
from src.data.quality import SAT_MASK_CNO_THRESHOLD, sat_mask


class _SeriesStub:
    """最小 pandas.Series 替身：仅实现 sat_mask 所需的比较运算。"""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)

    def __gt__(self, other: float) -> list[bool]:
        return [value > other for value in self.values]


class ContextTest(unittest.TestCase):
    """Context 结构与取值约束（§7.2、§7.4）。"""

    def test_components_follow_doc(self) -> None:
        self.assertEqual(CONTEXT_COMPONENTS, ("S", "Q", "O", "N", "H", "D"))

    def test_data_quality_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            Context(data_quality=1.5)

    def test_confidence_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            Context(confidence=-0.1)

    def test_unknown_component_raises(self) -> None:
        with self.assertRaises(KeyError):
            Context().component("X")

    def test_is_complete_requires_all_components(self) -> None:
        full = Context(S={"a": 1}, Q={"b": 1}, O={"c": 1}, N={"d": 1}, H={"e": 1}, D={"f": 1})
        self.assertTrue(full.is_complete)
        self.assertFalse(Context(S={"a": 1}).is_complete)

    def test_to_dict_roundtrip_keys(self) -> None:
        payload = Context(S={"a": 1}, confidence=0.9, data_quality=0.96).to_dict()
        for key in (*CONTEXT_COMPONENTS, "confidence", "data_quality", "timestamp"):
            self.assertIn(key, payload)

    def test_ablation_switch_returns_none(self) -> None:
        """§18.1 的“无 Context”对照组：编码器应返回 None 而非抛错。"""
        self.assertIsNone(ContextEncoder(enabled=False).encode())


class SatMaskTest(unittest.TestCase):
    """§5.4 Q3 的卫星有效性判据。"""

    def test_threshold_matches_doc(self) -> None:
        self.assertEqual(SAT_MASK_CNO_THRESHOLD, 0.5)

    def test_mask_marks_valid_satellites(self) -> None:
        mask = sat_mask(_SeriesStub([0.0, 0.5, 0.6, 45.0]))
        self.assertEqual(mask, [False, False, True, True])

    def test_missing_satellite_not_treated_as_low_signal(self) -> None:
        """C/N0 = 0（无卫星）必须被判为无效，避免误当成信号衰减。"""
        mask = sat_mask(_SeriesStub([0.0]))
        self.assertEqual(mask, [False])


if __name__ == "__main__":
    unittest.main()
