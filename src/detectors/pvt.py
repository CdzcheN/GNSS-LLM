"""S6 PVT/DOP 检测器：从导航解算层交叉验证异常。

对应开发文档
    §8.2 策略分类（S6 PVT/DOP检测）、§8.3 策略适用性矩阵（PVT 异常 → 首选 PVT/DOP，
    辅助观测一致性，目的“从导航解算层验证”）、§6.3（PVT 变化率、DOP 滑动统计、
    ``clkB`` / ``clkD`` 一阶差分）、§8.1 统一接口。

职责
    1. 依据精度指标（``hAcc`` / ``vAcc`` / ``tAcc``）与 DOP 相对基线的抬升给出判定；
    2. 依据钟差一阶差分（``clkB_diff`` / ``clkD_diff``）识别解算突变；
    3. 作为其他检测器的交叉验证方（§9.5 原则 B 互补信息优先）。

不做（边界）
    - 不做观测量层判据（归 S5）；
    - 不做导航解算本身（使用接收机输出的 PVT/DOP）；
    - 不做事件级结论（属 M7，§13）。

输入 / 输出
    输入：``data`` 为某历元的特征映射；``config`` 给出阈值
    输出：DetectionResult（§8.1）

证据字段（evidence）
    - ``hAcc`` / ``pDOP`` 等当前值与 ``*_vs_baseline`` 增量；
    - ``clkB_diff`` / ``clkD_diff``：钟差一阶差分（§6.3）；
    - ``acc_rise`` / ``dop_rise`` / ``clock_jump``：本次使用的阈值。

关键约束
    - 基线只使用历史行（由 ``src/features/navigation.py`` 以保证，§16.3）；
    - 阈值只来自 ``config``，默认值仅为可运行起点，须标定后回填（§20.3）；
    - 证据必须给出具体数值与单位语义（§2.2）。
"""

from __future__ import annotations

from typing import Any, Mapping

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus, numeric

#: 默认判据（**未标定**，仅作为可运行起点；正式阈值见 config.yaml 的 detectors.pvt）。
DEFAULT_ACC_RISE: float = 20.0
DEFAULT_DOP_RISE: float = 1.0
DEFAULT_CLOCK_JUMP: float = 100_000.0

#: 参与判定的精度列与其基线增量列。
ACCURACY_COLUMNS: tuple[str, ...] = ("hAcc", "vAcc", "tAcc")
DOP_COLUMNS: tuple[str, ...] = ("pDOP", "tDOP", "hDOP")
CLOCK_DIFF_COLUMNS: tuple[str, ...] = ("clkB_diff", "clkD_diff")


class PvtDetector(BaseDetector):
    """PVT/DOP 检测器（§8.2 S6）。"""

    detector_id = "pvt"
    supported_context = ("N",)
    version = "v1"

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """执行 PVT/DOP 检测（§8.1）。

        Args:
            data: 特征映射（需含 ``*_vs_baseline`` 与 ``clkB_diff`` 等派生列）。
            context: GNSS Context（读取数据质量）。
            config: 判据阈值，键为 ``acc_rise`` / ``dop_rise`` / ``clock_jump``。

        Returns:
            DetectionResult；派生特征缺失时返回 ``SKIPPED``。

        判据（满足其一即判为 Spoofing——导航解算与观测/卫星层不一致时更可能是欺骗）：
            1. 任一精度指标的相对基线抬升 ≥ ``acc_rise``；
            2. 任一 DOP 的相对基线抬升 ≥ ``dop_rise``；
            3. 钟差一阶差分的绝对值 ≥ ``clock_jump``。
        """
        settings = dict(config or {})
        acc_rise = float(settings.get("acc_rise", DEFAULT_ACC_RISE))
        dop_rise = float(settings.get("dop_rise", DEFAULT_DOP_RISE))
        clock_jump = float(settings.get("clock_jump", DEFAULT_CLOCK_JUMP))

        acc_values = {name: numeric(data, f"{name}_vs_baseline") for name in ACCURACY_COLUMNS}
        dop_values = {name: numeric(data, f"{name}_vs_baseline") for name in DOP_COLUMNS}
        clock_values = {name: numeric(data, name) for name in CLOCK_DIFF_COLUMNS}

        if all(value is None for value in (*acc_values.values(), *dop_values.values(), *clock_values.values())):
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={
                    "reason": "features_missing",
                    "required": [f"{name}_vs_baseline" for name in (*ACCURACY_COLUMNS, *DOP_COLUMNS)],
                },
                status=DetectorStatus.SKIPPED,
            )

        severity = 0.0
        for value in acc_values.values():
            if value is not None and acc_rise > 0:
                severity = max(severity, min(1.0, max(0.0, value) / acc_rise))
        for value in dop_values.values():
            if value is not None and dop_rise > 0:
                severity = max(severity, min(1.0, max(0.0, value) / dop_rise))
        for value in clock_values.values():
            if value is not None and clock_jump > 0:
                severity = max(severity, min(1.0, abs(value) / clock_jump))

        triggered = (
            any(value is not None and value >= acc_rise for value in acc_values.values())
            or any(value is not None and value >= dop_rise for value in dop_values.values())
            or any(value is not None and abs(value) >= clock_jump for value in clock_values.values())
        )

        evidence: dict[str, Any] = {
            "accuracy_vs_baseline": {name: value for name, value in acc_values.items()},
            "dop_vs_baseline": {name: value for name, value in dop_values.items()},
            "clock_diff": {name: value for name, value in clock_values.items()},
            "acc_rise": acc_rise,
            "dop_rise": dop_rise,
            "clock_jump": clock_jump,
            "severity": round(severity, 4),
        }
        if triggered:
            evidence["reason"] = "pvt_deviation"

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=AttackType.SPOOFING if triggered else AttackType.NORMAL,
            confidence=min(1.0, severity) if triggered else max(0.0, 1.0 - severity),
            evidence=evidence,
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )

    def describe(self):  # type: ignore[override]
        """返回注册表元信息（§8.4）。"""
        meta = super().describe()
        meta.input_schema = {
            "type": "navigation_features",
            "required": [f"{name}_vs_baseline" for name in ACCURACY_COLUMNS],
        }
        meta.output_schema = {"type": "DetectionResult"}
        meta.expected_latency_ms = 0.5
        meta.expected_cost = 1.0
        return meta
