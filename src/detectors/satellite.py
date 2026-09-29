"""S4 卫星状态检测器：可见星数量骤降与卫星层异常的识别。

对应开发文档
    §8.2 策略分类（S4 卫星状态检测）、§8.3 策略适用性矩阵（卫星状态异常 → 首选卫星检测，
    辅助 PVT，目的“判断卫星层异常”）、§6.3（可见卫星数量变化率）、§8.1 统一接口。

职责
    1. 依据可见卫星数（``NumSats``）与有效卫星数（``valid_sat_count``）的变化给出判定；
    2. 输出可复核的数值证据（当前值、基线、差值、阈值）；
    3. 为 S6 PVT 检测提供卫星层线索（§8.3 组合）。

不做（边界）
    - 不做 C/N0 幅度判定（归 S3）；
    - 不做导航解算结果判定（归 S6）；
    - 不做观测一致性分析（归 S5）。

输入 / 输出
    输入：``data`` 为某历元的特征映射；``config`` 给出阈值
    输出：DetectionResult（§8.1）

证据字段（evidence）
    - ``num_sats`` / ``valid_sat_count``：可见与有效卫星数；
    - ``sat_count_delta``：相对历史基线的变化；
    - ``sat_count_drop`` / ``min_sat_count``：本次使用的阈值。

关键约束
    - 卫星“消失”与“被遮挡”未在本层区分（需结合 C/N0 与几何，属后续改进，§2.2）；
    - 阈值只来自 ``config``；默认值仅为可运行起点，须标定后回填（§20.3）；
    - 变化量由 ``src/features/signal.py`` 以“不含未来行”的方式计算（§16.3）。
"""

from __future__ import annotations

from typing import Any, Mapping

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus, numeric

#: 默认判据（**未标定**，仅作为可运行起点；正式阈值见 config.yaml 的 detectors.satellite）。
DEFAULT_SAT_COUNT_DROP: float = -3.0
DEFAULT_MIN_SAT_COUNT: float = 4.0

FEATURE_SAT_COUNT_DELTA: str = "sat_count_delta"
FEATURE_NUM_SATS: str = "NumSats"
FEATURE_VALID_SAT_COUNT: str = "valid_sat_count"


class SatelliteDetector(BaseDetector):
    """卫星状态检测器（§8.2 S4）。"""

    detector_id = "satellite"
    supported_context = ("Q", "O")
    version = "v1"

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """执行卫星状态检测（§8.1）。

        Args:
            data: 特征映射（需含 ``sat_count_delta`` 与卫星数量列）。
            context: GNSS Context（读取数据质量）。
            config: 判据阈值，键为 ``sat_count_drop`` / ``min_sat_count``。

        Returns:
            DetectionResult；特征缺失时返回 ``SKIPPED``。

        判据：卫星数下降达到阈值 **或** 有效卫星数低于下限 ⇒ 判为 Jamming
        （卫星层与压制在特征上高度重合，欺骗场景由 S5/S6 补充）。
        """
        settings = dict(config or {})
        drop_threshold = float(settings.get("sat_count_drop", DEFAULT_SAT_COUNT_DROP))
        min_sat_count = float(settings.get("min_sat_count", DEFAULT_MIN_SAT_COUNT))

        delta = numeric(data, FEATURE_SAT_COUNT_DELTA)
        num_sats = numeric(data, FEATURE_NUM_SATS)
        valid_sats = numeric(data, FEATURE_VALID_SAT_COUNT)

        observed = valid_sats if valid_sats is not None else num_sats
        if delta is None and observed is None:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={
                    "reason": "features_missing",
                    "required": [FEATURE_SAT_COUNT_DELTA, FEATURE_VALID_SAT_COUNT],
                },
                status=DetectorStatus.SKIPPED,
            )

        severity = 0.0
        if delta is not None and drop_threshold < 0:
            severity = max(severity, min(1.0, abs(min(delta, 0.0)) / abs(drop_threshold)))
        if observed is not None and min_sat_count > 0:
            severity = max(severity, min(1.0, max(0.0, (min_sat_count - observed) / min_sat_count)))

        abnormal = severity > 0.0 and (
            (delta is not None and delta <= drop_threshold)
            or (observed is not None and observed < min_sat_count)
        )

        evidence = {
            "num_sats": num_sats,
            "valid_sat_count": valid_sats,
            "sat_count_delta": delta,
            "sat_count_drop": drop_threshold,
            "min_sat_count": min_sat_count,
            "severity": round(severity, 4),
        }
        if abnormal:
            evidence["reason"] = "satellite_count_drop"

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=AttackType.JAMMING if abnormal else AttackType.NORMAL,
            confidence=min(1.0, severity) if abnormal else max(0.0, 1.0 - severity),
            evidence=evidence,
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )

    def describe(self):  # type: ignore[override]
        """返回注册表元信息（§8.4）。"""
        meta = super().describe()
        meta.input_schema = {"type": "satellite_features", "required": [FEATURE_SAT_COUNT_DELTA]}
        meta.output_schema = {"type": "DetectionResult"}
        meta.expected_latency_ms = 0.5
        meta.expected_cost = 1.0
        return meta
