"""S2 频谱检测器：从频谱形态识别明显射频异常。

对应开发文档
    §8.2 策略分类（S2 频谱检测）、§8.3 策略适用性矩阵（频谱异常明显 → 首选频谱检测，
    辅助 C/N0，目的“快速发现射频异常”）、§8.1 统一接口、§5.3 模态 B。

职责
    1. 依据频谱统计量与形态判据输出射频异常判定；
    2. 为 Context 的 S_t（信号/频谱状态）提供解释性证据；
    3. 与 C/N0 检测器形成“频谱 + 信号质量”的首选/辅助组合（§8.3）。

不做（边界）
    - 不做原始 I/Q 全链路处理（§1.5 明确不在范围内）；
    - 不做卫星层与观测层判定（归 S4/S5）；
    - 不直接给出事件级结论（属 M6/M7，§12、§13）。

输入 / 输出
    输入：频谱统计特征或谱表示（M2 features/spectrum.py 的输出）
    输出：DetectionResult（§8.1）

证据字段（evidence，计划）
    - ``peak_freq`` / ``bandwidth``：异常峰的位置与宽度；
    - ``in_band_ratio``：带内能量占比变化；
    - ``agc_correlation``：与 AGC 变化的联动关系（用于排除增益变化，§2.2）。

关键约束
    - 频谱判据必须与 AGC 联动核对，避免把接收机自动增益变化误判为干扰；
    - 阈值需在标注数据上标定并记录在实验日志（§20.3），不得凭经验硬编码；
    - 判定必须可解释：给出具体频点与能量依据（§2.2）。

待实现
    - 判据标定：在 1221 干扰事件区间上确定阈值/形态判据；
    - 与 S3 C/N0 检测器的互补调用逻辑在 §9.5 原则 B 下验证。
"""

from __future__ import annotations

from typing import Any, Mapping

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus

#: 与 features/spectrum.py 的定长谱表示维度保持一致（§5.3 模态 B）。
SPECTRUM_EMBEDDING_DIM: int = 128


class SpectrumDetector(BaseDetector):
    """频谱检测器（§8.2 S2）。"""

    detector_id = "spectrum"
    supported_context = ("S",)
    version = "v0"

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """执行频谱检测（§8.1）。

        Args:
            data: 频谱统计特征或谱表示。
            context: GNSS Context（使用 S_t 分量）。
            config: 判据与阈值配置。

        Returns:
            DetectionResult。

        Raises:
            NotImplementedError: 判据需在标注数据上标定后实现（见“待实现”）。

        对应开发文档：§8.2 S2、§8.3。
        """
        raise NotImplementedError("TODO(§8.2 S2): 频谱判据标定后实现")

    def describe(self):  # type: ignore[override]
        """返回注册表元信息，标注预期的时延/开销等级（§8.4、§9.5 原则 A）。"""
        meta = super().describe()
        meta.input_schema = {"type": "spectrum_features", "dim_optional": SPECTRUM_EMBEDDING_DIM}
        meta.output_schema = {"type": "DetectionResult", "attack_type": "jamming|normal"}
        meta.expected_latency_ms = 1.0
        meta.expected_cost = 1.0
        meta.reliability = 0.0  # 待 §17.4 指标回填
        return meta


#: 该模块未使用但保留的类型引用，避免误删对三态枚举的依赖（供判据实现使用）。
_ATTACK_TYPE_REF = AttackType
_STATUS_REF = DetectorStatus
