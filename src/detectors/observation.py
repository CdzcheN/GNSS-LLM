"""S5 观测一致性检测器：从伪距残差异常识别潜在欺骗。

对应开发文档
    §8.2 策略分类（S5 观测一致性检测）、§8.3 策略适用性矩阵（观测量不一致 → 首选观测一致性，
    辅助深度时序，目的“识别潜在欺骗”）、§6.3（伪距/载波/多普勒一致性）、§8.1 统一接口。

职责
    1. 依据伪距残差的超限个数、最大值与相对基线抬升给出判定；
    2. 输出按 PRN 可追溯的超限明细（§2.2）；
    3. 为深度时序检测器（S7）提供互补输入（§9.5 原则 C）。

不做（边界）
    - 不做频谱与 C/N0 层判定（归 S2/S3）；
    - 不做最终欺骗标签判定（属 M6 融合，§12）；
    - 不做多普勒/载波一致性：当前数据集**未提供**这两类列（见文件末说明）。

输入 / 输出
    输入：``data`` 为某历元的完整特征映射（含 ``Res_Gxx`` 原始残差）；
          ``config`` 给出阈值
    输出：DetectionResult（§8.1）

证据字段（evidence）
    - ``residual_outliers``：超限残差的 PRN 与数值明细（最多 ``max_evidence`` 条）；
    - ``res_outlier_count`` / ``res_valid_max`` / ``res_delta``：统计量；
    - ``residual_threshold`` / ``outlier_count`` / ``baseline_rise``：本次使用的阈值。

关键约束
    - 只统计**有效卫星**（C/N0 > 0.5）的残差，避免把无观测当成零残差（§5.4 Q3）；
    - 阈值只来自 ``config``，默认值仅为可运行起点，须标定后回填（§20.3）；
    - 证据必须列出具体 PRN 与残差值（§2.2）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import schema
from src.data.quality import SAT_MASK_CNO_THRESHOLD
from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus, numeric

#: 默认判据（**未标定**，仅作为可运行起点；正式阈值见 config.yaml 的 detectors.observation）。
DEFAULT_RESIDUAL_THRESHOLD: float = 15.0
DEFAULT_OUTLIER_COUNT: float = 2.0
DEFAULT_BASELINE_RISE: float = 5.0

#: 证据中最多列出的超限明细条数（避免证据过长）。
DEFAULT_MAX_EVIDENCE: int = 5

FEATURE_OUTLIER_COUNT: str = "res_outlier_count"
FEATURE_RES_MAX: str = "res_valid_max"
FEATURE_RES_DELTA: str = "res_delta"

#: 主判据特征：**原始列 ``MaxRes``**。
#: 标定显示其区分度（AUC 0.859）明显强于派生的 ``res_valid_max``（0.746）——
#: 欺骗期间单个卫星的残差可达数百米，而逐历元有效卫星的残差最大值反而被平均掉。
FEATURE_MAX_RES: str = "MaxRes"


def residual_outliers(
    data: Mapping[str, Any] | None,
    threshold: float,
    cno_threshold: float = SAT_MASK_CNO_THRESHOLD,
    limit: int = DEFAULT_MAX_EVIDENCE,
    res_columns: Sequence[str] = schema.RES_COLUMNS,
    cno_columns: Sequence[str] = schema.CNO_COLUMNS,
) -> list[dict[str, Any]]:
    """列出超限残差的 PRN 与数值明细（供 evidence 使用，§2.2）。

    Args:
        data: 某历元的特征映射（含 ``Res_Gxx`` 与 ``CNO_Gxx``）。
        threshold: 超限阈值。
        cno_threshold: 卫星有效性阈值（§5.4 Q3）。
        limit: 最多返回条数。
        res_columns: 残差列组。
        cno_columns: C/N0 列组。

    Returns:
        按残差降序排列的 ``{"prn", "residual"}`` 列表。
    """
    records: list[dict[str, Any]] = []
    for res_column, cno_column in zip(res_columns, cno_columns):
        residual = numeric(data, res_column)
        cno = numeric(data, cno_column)
        if residual is None or cno is None or cno <= float(cno_threshold):
            continue
        if residual > float(threshold):
            records.append({"prn": res_column.split("_")[-1], "residual": round(residual, 4)})
    records.sort(key=lambda item: item["residual"], reverse=True)
    return records[:limit] if limit > 0 else records


class ObservationDetector(BaseDetector):
    """观测一致性检测器（§8.2 S5）。"""

    detector_id = "observation"
    supported_context = ("O",)
    version = "v1"

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """执行观测一致性检测（§8.1）。

        Args:
            data: 特征映射（建议传入整行特征，以便生成 PRN 明细）。
            context: GNSS Context（读取数据质量）。
            config: 判据阈值，键为 ``residual_threshold`` / ``outlier_count`` / ``baseline_rise``。

        Returns:
            DetectionResult；统计特征缺失时返回 ``SKIPPED``。

        判据（满足其一即判为 Spoofing）：
            1. 超限残差个数 ≥ ``outlier_count``；
            2. 最大有效残差 > ``residual_threshold``；
            3. 残差均值相对基线抬升 ≥ ``baseline_rise``。
        """
        settings = dict(config or {})
        threshold = float(settings.get("residual_threshold", DEFAULT_RESIDUAL_THRESHOLD))
        required_outliers = float(settings.get("outlier_count", DEFAULT_OUTLIER_COUNT))
        baseline_rise = float(settings.get("baseline_rise", DEFAULT_BASELINE_RISE))
        cno_threshold = float(settings.get("cno_threshold", SAT_MASK_CNO_THRESHOLD))
        max_evidence = int(settings.get("max_evidence", DEFAULT_MAX_EVIDENCE))

        outlier_count = numeric(data, FEATURE_OUTLIER_COUNT)
        res_max = numeric(data, FEATURE_RES_MAX)
        res_delta = numeric(data, FEATURE_RES_DELTA)
        max_res = numeric(data, FEATURE_MAX_RES)

        if outlier_count is None and res_max is None and res_delta is None and max_res is None:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={
                    "reason": "features_missing",
                    "required": [FEATURE_MAX_RES, FEATURE_OUTLIER_COUNT, FEATURE_RES_MAX],
                },
                status=DetectorStatus.SKIPPED,
            )

        severity = 0.0
        if max_res is not None and threshold > 0:
            severity = max(severity, min(1.0, max_res / threshold))
        if outlier_count is not None and required_outliers > 0:
            severity = max(severity, min(1.0, outlier_count / required_outliers))
        if res_max is not None and threshold > 0:
            severity = max(severity, min(1.0, res_max / threshold))
        if res_delta is not None and baseline_rise > 0:
            severity = max(severity, min(1.0, max(0.0, res_delta) / baseline_rise))

        triggered = (
            (max_res is not None and max_res >= threshold)
            or (outlier_count is not None and outlier_count >= required_outliers)
            or (res_max is not None and res_max > threshold)
            or (res_delta is not None and res_delta >= baseline_rise)
        )

        evidence: dict[str, Any] = {
            "MaxRes": max_res,
            "res_outlier_count": outlier_count,
            "res_valid_max": res_max,
            "res_delta": res_delta,
            "residual_threshold": threshold,
            "outlier_count": required_outliers,
            "baseline_rise": baseline_rise,
            "severity": round(severity, 4),
        }

        # 从原始行构造 PRN 明细（数据缺失时为空列表）
        details = residual_outliers(
            data, threshold=threshold, cno_threshold=cno_threshold, limit=max_evidence
        )
        if details:
            evidence["residual_outliers"] = details

        if triggered:
            evidence["reason"] = "residual_inconsistency"
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.SPOOFING,
                confidence=min(1.0, severity),
                evidence=evidence,
                data_quality=None if context is None else getattr(context, "data_quality", None),
                status=DetectorStatus.OK,
            )

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=AttackType.NORMAL,
            confidence=max(0.0, 1.0 - severity),
            evidence=evidence,
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )

    def describe(self):  # type: ignore[override]
        """返回注册表元信息（§8.4）。"""
        meta = super().describe()
        meta.input_schema = {
            "type": "observation_features",
            "required": [FEATURE_OUTLIER_COUNT, FEATURE_RES_MAX],
        }
        meta.output_schema = {"type": "DetectionResult", "attack_type": "spoofing|normal"}
        meta.expected_latency_ms = 1.0
        meta.expected_cost = 1.5
        return meta
