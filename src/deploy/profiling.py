"""模型侧性能测量：参数量、推理时延与内存占用（§19.1）。

对应开发文档
    §19.1 模型部署目标（参数量、模型文件大小、单窗口推理时延、内存占用、INT8 量化）、
    §1.4 G8 轻量化目标（深度检测核心模型参数量 < 10⁶）、§17.5 系统级指标。

职责
    1. 测量单窗口推理时延（含预热与分位数统计），无需第三方库；
    2. 统计模型参数量与参数量预算校验（§1.4 G8）；
    3. 汇总为 ProfileReport，供 §19 部署评估与 §20.3 实验记录使用。

不做（边界）
    - 不做量化本身（→ ``src/deploy/quantize.py``）；
    - 不在测量中改变模型状态（必须处于推理模式，由调用方保证）；
    - 不报告无法追溯的单一数字：时延必须给出重复次数与分位数。

输入 / 输出
    输入：可调用对象（模型前向）与重复次数
    输出：ProfileReport（时延分位数、参数量、体积、内存）

关键约束
    - 时延测量必须包含预热轮，避免把首次编译/分配开销计入稳态时延；
    - 参数量必须与 §1.4 G8 的预算比较并给出结论，不得只报数字；
    - 仅在 CPU 稳定测量时延（GPU 需另行同步与标注，属后续扩展）。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from src.detectors.deep_temporal import PARAMETER_BUDGET

#: 默认测量重复次数与预热次数（稳态时延统计）。
DEFAULT_REPEATS: int = 100
DEFAULT_WARMUP: int = 5

#: 参数量单位换算（百万）。
PARAMETERS_PER_MILLION: float = 1e6


@dataclass(slots=True)
class LatencyStats:
    """推理时延统计（毫秒）。"""

    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(slots=True)
class ProfileReport:
    """部署性能报告（§19.1）。

    Attributes:
        latency: 时延统计。
        parameters: 模型参数量；``None`` 表示未测量（如无 torch）。
        parameter_budget: 参数量预算（§1.4 G8）。
        within_budget: 是否满足参数量预算。
        model_size_mb: 模型文件体积（MB），由调用方给出。
        memory_mb: 进程内存占用（MB），由调用方给出。
        notes: 备注（例如测量设备与线程数）。
    """

    latency: LatencyStats | None = None
    parameters: int | None = None
    parameter_budget: int = PARAMETER_BUDGET
    within_budget: bool | None = None
    model_size_mb: float | None = None
    memory_mb: float | None = None
    notes: str = ""

    def to_dict(self) -> Mapping[str, Any]:
        """转为可写入实验记录的映射（§20.3）。"""
        payload = asdict(self)
        if self.parameters is not None:
            payload["parameters_million"] = self.parameters / PARAMETERS_PER_MILLION
        return payload


def measure_latency(
    forward: Callable[[], Any],
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> LatencyStats:
    """测量可调用对象的稳态执行时延。

    Args:
        forward: 单次前向调用（应已完成输入准备，避免把数据搬运计入时延）。
        repeats: 有效测量次数（必须 >= 1）。
        warmup: 预热次数（>= 0）。

    Returns:
        LatencyStats（毫秒）。

    Raises:
        ValueError: ``repeats`` < 1 或 ``warmup`` < 0。
    """
    if repeats < 1:
        raise ValueError(f"repeats 必须 >= 1，实际为 {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup 不能为负：{warmup}")

    for _ in range(warmup):
        forward()

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        forward()
        samples.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples)
    # 最近秩法计算 p95，避免插值带来的口径歧义
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    return LatencyStats(
        count=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        median_ms=median(ordered),
        p95_ms=ordered[p95_index],
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def count_parameters(model: Any) -> int:
    """统计模型参数量（§1.4 G8）。

    Args:
        model: 支持 ``parameters()`` 的模型对象（通常为 ``torch.nn.Module``）。

    Returns:
        可训练参数量。

    Raises:
        TypeError: 模型不支持 ``parameters()``。
    """
    if not hasattr(model, "parameters"):
        raise TypeError(f"模型需提供 parameters() 方法，收到 {type(model).__name__}")
    return int(sum(parameter.numel() for parameter in model.parameters()))


def check_parameter_budget(
    parameters: int,
    budget: int = PARAMETER_BUDGET,
) -> Mapping[str, Any]:
    """校验参数量是否满足轻量化目标（§1.4 G8）。

    Args:
        parameters: 模型参数量。
        budget: 预算上限（默认 10⁶）。

    Returns:
        含 ``ok``、``parameters``、``budget``、``utilization`` 的映射。
    """
    if parameters < 0:
        raise ValueError(f"参数量不能为负：{parameters}")
    return {
        "ok": parameters < budget,
        "parameters": parameters,
        "budget": budget,
        "utilization": parameters / budget if budget else 0.0,
    }


def profile(
    forward: Callable[[], Any] | None = None,
    model: Any | None = None,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    model_size_mb: float | None = None,
    memory_mb: float | None = None,
    notes: str = "",
) -> ProfileReport:
    """汇总一次部署性能测量（§19.1）。

    Args:
        forward: 单次前向调用；``None`` 表示跳过时延测量。
        model: 模型对象；``None`` 表示跳过参数量统计。
        repeats: 时延测量次数。
        warmup: 预热次数。
        model_size_mb: 模型文件体积（MB）。
        memory_mb: 内存占用（MB）。
        notes: 备注。

    Returns:
        ProfileReport。
    """
    latency = measure_latency(forward, repeats, warmup) if forward is not None else None
    parameters = count_parameters(model) if model is not None else None
    within = check_parameter_budget(parameters)["ok"] if parameters is not None else None

    return ProfileReport(
        latency=latency,
        parameters=parameters,
        within_budget=within,
        model_size_mb=model_size_mb,
        memory_mb=memory_mb,
        notes=notes,
    )


#: 供调用方直接引用的系统级指标键（§17.5）。
SYSTEM_METRIC_KEYS: Sequence[str] = ("end_to_end_latency", "memory", "parameters")
