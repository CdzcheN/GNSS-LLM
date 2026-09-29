"""M4/M5 检测器注册表：把检测器收敛为“可调用工具”。

对应开发文档
    §8.4 Detector Registry、§9.1 智能体定位（只面对可调用工具而非底层代码）、
    §9.6 实现层级（工具集有限且受约束）、§2.3 动态选择必须可验证。

职责
    1. 登记检测器及其元信息（§8.4 的 8 个字段）；
    2. 提供按 id / 上下文分量查询能力，支撑 §8.3 策略适用性匹配；
    3. 输出“工具视图”（``as_tools()``），供 Agent 层（Level 3）消费（§9.6）。

不做（边界）
    - 不执行检测（执行由 executor 负责）；
    - 不做策略选择（由 policy/planner 负责，§9）；
    - 不修改检测器返回结果（§2.4）。

输入 / 输出
    输入：BaseDetector 实例
    输出：DetectorMeta 查询结果、工具视图列表

关键约束
    - 重复注册同一 ``detector_id`` 必须显式失败，避免静默覆盖导致实验不可复现（§20.3）；
    - 注册表是“无状态查询”组件，不做缓存与隐式排序，保证行为确定（§20.4）；
    - 工具视图必须暴露时延与开销，让编排层能执行 §9.5 原则 A（低成本优先）。
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

from src.detectors.base import BaseDetector, DetectorMeta


class DetectorRegistry:
    """检测器注册表（§8.4）。"""

    def __init__(self) -> None:
        """创建空注册表。"""
        self._detectors: dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector) -> DetectorMeta:
        """注册一个检测器。

        Args:
            detector: 检测器实例（须继承 BaseDetector）。

        Returns:
            该检测器的注册元信息。

        Raises:
            TypeError: 传入对象不是 BaseDetector。
            ValueError: ``detector_id`` 已被占用。
        """
        if not isinstance(detector, BaseDetector):
            raise TypeError(f"只能注册 BaseDetector 实例，收到 {type(detector).__name__}")
        detector_id = detector.detector_id
        if detector_id in self._detectors:
            raise ValueError(f"检测器 {detector_id!r} 已注册，禁止静默覆盖（§20.3 可复现）")
        self._detectors[detector_id] = detector
        return detector.describe()

    def unregister(self, detector_id: str) -> None:
        """移除检测器。

        Args:
            detector_id: 检测器标识。

        Raises:
            KeyError: 未注册。
        """
        del self._detectors[detector_id]

    def get(self, detector_id: str) -> BaseDetector:
        """按 id 取检测器实例。

        Args:
            detector_id: 检测器标识。

        Returns:
            检测器实例。

        Raises:
            KeyError: 未注册（调用方应视为编排错误，§2.3）。
        """
        try:
            return self._detectors[detector_id]
        except KeyError as exc:
            raise KeyError(f"未注册的检测器：{detector_id!r}，已注册 {sorted(self._detectors)}") from exc

    def meta(self, detector_id: str) -> DetectorMeta:
        """按 id 取元信息（§8.4）。"""
        return self.get(detector_id).describe()

    def ids(self) -> list[str]:
        """返回已注册检测器 id 列表（排序稳定，便于实验记录）。"""
        return sorted(self._detectors)

    def __iter__(self) -> Iterator[BaseDetector]:
        """按 id 升序迭代检测器。"""
        for detector_id in self.ids():
            yield self._detectors[detector_id]

    def __len__(self) -> int:
        return len(self._detectors)

    def by_context(self, component: str) -> list[str]:
        """列出适用于某个上下文分量（S/Q/O/N/H/D）的检测器（§7.2、§8.3）。

        Args:
            component: 上下文分量名。

        Returns:
            检测器 id 列表。
        """
        return [
            detector_id
            for detector_id in self.ids()
            if component in self._detectors[detector_id].supported_context
        ]

    def as_tools(self) -> list[Mapping[str, Any]]:
        """输出工具视图，供 Agent 层（§9.6 Level 3）编排使用。

        Returns:
            每个元素描述一个可调用工具：id、适用上下文、预期时延与开销。
        """
        tools: list[Mapping[str, Any]] = []
        for detector in self:
            meta = detector.describe()
            tools.append(
                {
                    "name": meta.detector_id,
                    "supported_context": list(meta.supported_context),
                    "expected_latency_ms": meta.expected_latency_ms,
                    "expected_cost": meta.expected_cost,
                    "reliability": meta.reliability,
                    "version": meta.version,
                }
            )
        return tools
