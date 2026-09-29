"""训练可复现性：固定随机种子与运行时状态记录。

对应开发文档
    §20.4 固定随机种子（基线 seed = 42；研究性实验可多 seed 重复并报告均值与标准差）、
    §2.5 时间序列禁止随机泄漏、§16.3 防泄漏规范、§20.3 实验记录规范。

职责
    1. 一次性设定 Python / NumPy / PyTorch 的随机种子并回读生效结果；
    2. 为多 seed 重复实验生成种子序列（同一基线 seed 派生，避免人工挑选）；
    3. 把生效状态写入实验记录（§20.3 的 seed 字段）。

不做（边界）
    - 不保证跨设备、跨版本的位级一致（取决于算子实现，属 §19 部署范畴）；
    - 不修改数据划分（划分由 §16.2 的固定协议决定，随机打散被 §2.5 禁止）。

输入 / 输出
    输入：seed 整数（默认 42）
    输出：各框架实际生效的种子映射

关键约束
    - 基线默认 ``seed = 42``（§20.4），训练代码不得另取未记录的随机源；
    - NumPy / PyTorch 采用惰性导入：未安装时只设定可用框架，不因此报错；
    - 多 seed 实验的种子必须可复现（本节用确定性派生而非随机生成）。
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping, Sequence

#: 基线默认随机种子（§20.4）。
DEFAULT_SEED: int = 42

#: 多 seed 实验的默认重复次数（研究性实验报告均值与标准差，§20.4）。
DEFAULT_REPEATS: int = 3

#: 派生种子的固定步长，保证同一基线 seed 得到稳定序列。
_SEED_STRIDE: int = 1000


def set_global_seed(seed: int = DEFAULT_SEED) -> Mapping[str, Any]:
    """设定全局随机种子。

    Args:
        seed: 随机种子。

    Returns:
        实际生效的种子映射，键为 ``python`` / ``numpy`` / ``torch``；
        对应框架未安装时值为 ``None``。

    Note:
        ``PYTHONHASHSEED`` 只影响本进程后续行为；如需完全生效应在解释器启动前设置。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    applied: dict[str, Any] = {"python": seed}

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - 依赖环境相关
        applied["numpy"] = None
    else:
        np.random.seed(seed)
        applied["numpy"] = seed

    try:
        import torch
    except ImportError:  # pragma: no cover - 依赖环境相关
        applied["torch"] = None
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - 需 GPU 环境
            torch.cuda.manual_seed_all(seed)
        applied["torch"] = seed

    return applied


def seed_sequence(
    base_seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> Sequence[int]:
    """生成多 seed 实验的种子序列（确定性派生）。

    Args:
        base_seed: 基线种子（§20.4 默认 42）。
        repeats: 重复次数（必须为正）。

    Returns:
        以 ``base_seed`` 为首、按固定步长派生的种子元组。

    Raises:
        ValueError: ``repeats`` 小于 1。
    """
    if repeats < 1:
        raise ValueError(f"repeats 必须 >= 1，实际为 {repeats}")
    return tuple(base_seed + index * _SEED_STRIDE for index in range(repeats))


def describe_environment() -> Mapping[str, Any]:
    """收集运行环境信息，用于实验记录与结果复现。

    Returns:
        含 Python 版本与关键库版本（numpy / torch / pandas，缺失记为 ``None``）的映射。
    """
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            info[name] = None
        else:
            info[name] = getattr(module, "__version__", "unknown")
    return info
