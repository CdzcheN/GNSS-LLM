"""S3 C/N0 检测器：通过载噪比整体下降判断压制干扰。

对应开发文档
    §8.2 策略分类（S3 C/N0检测）、§8.3 策略适用性矩阵（C/N0 整体下降 → 首选 C/N0 检测，
    辅助卫星状态，目的“判断压制影响”）、§5.4 Q3（sat_mask）、§8.1 统一接口。

职责
    1. 依据窗口内 C/N0 相对基线的下降量与有效卫星数给出压制类判定；
    2. 输出可复核的数值证据（下降 dB、有效星数、受影响卫星）；
    3. 与 S2 频谱检测互补（§9.5 原则 B：信息来源不同的检测器）。

不做（边界）
    - 不做频谱层判定（归 S2；模态 B 数据未提供）；
    - 不做卫星几何/可见性判定（归 S4）；
    - 不修改 C/N0 原始值，只在其上做统计。

输入 / 输出
    输入：``data`` 为某历元的特征映射（须含 ``cn0_delta_db``、``cn0_valid_count`` 等）；
          ``config`` 给出判据阈值
    输出：DetectionResult（§8.1）

证据字段（evidence）
    - ``cn0_valid_mean`` / ``cn0_baseline`` / ``cn0_delta_db``：当前均值与基线及差值；
    - ``cn0_valid_count``：有效卫星数；
    - ``drop_threshold_db`` / ``min_valid_sat``：本次使用的阈值（便于复核配置）；

关键约束
    - 必须使用 ``sat_mask`` 之后的统计（无卫星历元不会被当作低 C/N0，§5.4 Q3）；
    - 阈值只能来自 ``config``：本文件的默认值只是**可运行起点**，正式值须由
      验证集标定后写入 ``config.yaml``（§20.3 可追溯）；
    - 判据必须可解释：置信度由“超标程度”线性映射，不使用黑盒打分（§2.2）。
"""

from __future__ import annotations

from typing import Any, Mapping

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus, numeric

#: 默认判据（**未标定**，仅作为可运行起点；正式阈值见 config.yaml 的 detectors.cno）。
DEFAULT_CN0_DROP_DB: float = -3.0
DEFAULT_MIN_VALID_SAT: float = 4.0

#: 使用的特征列名（由 src/features/signal.py 产出）。
FEATURE_CN0_DELTA: str = "cn0_delta_db"
FEATURE_CN0_MEAN: str = "cn0_valid_mean"
FEATURE_CN0_BASELINE: str = "cn0_baseline"
FEATURE_CN0_COUNT: str = "cn0_valid_count"


class CnoDetector(BaseDetector):
    """C/N0 检测器（§8.2 S3）。"""

    detector_id = "cno"
    supported_context = ("Q", "S")
    version = "v1"

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """执行 C/N0 检测（§8.1）。

        Args:
            data: 特征映射（需含 ``cn0_delta_db`` 等列）。
            context: GNSS Context（用于读取数据质量）。
            config: 判据阈值，键为 ``cn0_drop_db`` / ``min_valid_sat``。

        Returns:
            DetectionResult；特征缺失时返回 ``SKIPPED``。

        判据：C/N0 下降达到阈值 **或** 有效卫星数低于下限 ⇒ 判为 Jamming；
        置信度取两项“超标程度”的较大者。
        """
        settings = dict(config or {})
        drop_threshold = float(settings.get("cn0_drop_db", DEFAULT_CN0_DROP_DB))
        min_valid_sat = float(settings.get("min_valid_sat", DEFAULT_MIN_VALID_SAT))

        delta = numeric(data, FEATURE_CN0_DELTA)
        count = numeric(data, FEATURE_CN0_COUNT)
        mean = numeric(data, FEATURE_CN0_MEAN)
        baseline = numeric(data, FEATURE_CN0_BASELINE)

        if delta is None and count is None:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={"reason": "features_missing", "required": [FEATURE_CN0_DELTA, FEATURE_CN0_COUNT]},
                status=DetectorStatus.SKIPPED,
            )

        # 两项严重度：C/N0 下降幅度、卫星数不足程度（均归一到 [0, 1]）
        severity = 0.0
        if delta is not None and drop_threshold < 0:
            severity = max(severity, min(1.0, abs(min(delta, 0.0)) / abs(drop_threshold)))
        if count is not None and min_valid_sat > 0:
            severity = max(severity, min(1.0, max(0.0, (min_valid_sat - count) / min_valid_sat)))

        jamming = severity > 0.0 and (
            (delta is not None and delta <= drop_threshold)
            or (count is not None and count < min_valid_sat)
        )

        evidence = {
            "cn0_valid_mean": mean,
            "cn0_baseline": baseline,
            "cn0_delta_db": delta,
            "cn0_valid_count": count,
            "drop_threshold_db": drop_threshold,
            "min_valid_sat": min_valid_sat,
            "severity": round(severity, 4),
        }
        if jamming:
            evidence["reason"] = "cn0_drop_or_satellite_loss"

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=AttackType.JAMMING if jamming else AttackType.NORMAL,
            confidence=min(1.0, severity) if jamming else max(0.0, 1.0 - severity),
            evidence=evidence,
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )

    def describe(self):  # type: ignore[override]
        """返回注册表元信息（§8.4）。"""
        meta = super().describe()
        meta.input_schema = {"type": "cn0_features", "required": [FEATURE_CN0_DELTA, FEATURE_CN0_COUNT]}
        meta.output_schema = {"type": "DetectionResult", "attack_type": "jamming|normal"}
        meta.expected_latency_ms = 0.5
        meta.expected_cost = 1.0
        return meta
