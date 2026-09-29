"""训练配置、早停状态机与训练循环接口。

对应开发文档
    §10.2 默认模型配置（LSTM 2 层、hidden 64、窗口 60 s、3 类）、§10.5 类别不平衡、
    §16 实验设计（划分与防泄漏）、§22 风险管理（深度模型过拟合 → 早停、正则、跨事件测试）、
    §20.3 实验记录规范、§20.4 固定随机种子。

职责
    1. 描述训练配置（轮数、批大小、学习率、早停耐心值、监控指标）；
    2. 提供 EarlyStopping 状态机——纯 Python 逻辑，可独立测试；
    3. 提供 Trainer 的训练 / 验证循环接口。

不做（边界）
    - 不实现模型结构（属 ``src/detectors/deep_temporal.py``）；
    - 不写实验记录（属 ``src/train/experiment.py``）；
    - 不做超参搜索（Optuna 为可选项，§20.1）。

输入 / 输出
    输入：TrainingConfig 与训练/验证数据
    输出：训练结果摘要（最优轮次、监控指标、checkpoint 路径）

关键约束
    - 早停必须依据**验证集**指标，不得使用测试集或 hold-out（§16.3）；
    - 监控指标默认 ``val_loss``（§22：跨事件掉点是过拟合信号）；
    - 训练随机性必须由 §20.4 的固定种子控制，不在本模块另设随机源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.detectors.base import AttackType
from src.train.reproducibility import DEFAULT_SEED

#: 早停监控方向。
MONITOR_MODES: tuple[str, ...] = ("min", "max")


@dataclass(slots=True)
class TrainingConfig:
    """训练超参与训练过程控制（§10.2、§22）。

    Attributes:
        epochs: 最大训练轮数。
        batch_size: 批大小。
        learning_rate: 学习率。
        early_stop_patience: 验证指标连续无改善的容忍轮数（§22 过拟合风险应对）。
        early_stop_min_delta: 视为“改善”的最小变化量。
        monitor: 早停监控指标名（默认 ``val_loss``）。
        monitor_mode: ``"min"`` 或 ``"max"``。
        seed: 随机种子（§20.4 基线 42）。
        window_s: 输入窗口（§6.2）。
        num_classes: 分类头类别数（§1.3 三态）。
        gradient_clip: 梯度裁剪阈值；``None`` 表示不裁剪。
        l2_weight_decay: L2 正则系数（§22 正则手段之一）。
    """

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    seed: int = DEFAULT_SEED
    window_s: int = 60
    num_classes: int = 3
    gradient_clip: float | None = 1.0
    l2_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        """校验超参合法性。"""
        if self.epochs <= 0:
            raise ValueError(f"epochs 必须为正：{self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正：{self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须为正：{self.learning_rate}")
        if self.early_stop_patience < 0:
            raise ValueError(f"early_stop_patience 不能为负：{self.early_stop_patience}")
        if self.monitor_mode not in MONITOR_MODES:
            raise ValueError(f"monitor_mode 需属于 {MONITOR_MODES}，实际为 {self.monitor_mode!r}")
        if self.num_classes != len(AttackType):
            raise ValueError(
                f"num_classes 必须等于三态类别数 {len(AttackType)}（§1.3），实际为 {self.num_classes}"
            )


@dataclass(slots=True)
class EarlyStopping:
    """早停状态机（§22）。

    只维护“是否改善”的判定与计数，不接触数据与模型，因此可独立单测。
    """

    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"
    best: float | None = None
    best_step: int = 0
    counter: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        """校验参数。"""
        if self.patience < 0:
            raise ValueError(f"patience 不能为负：{self.patience}")
        if self.mode not in MONITOR_MODES:
            raise ValueError(f"mode 需属于 {MONITOR_MODES}，实际为 {self.mode!r}")

    def is_improvement(self, value: float) -> bool:
        """判断给定指标值是否构成改善。

        Args:
            value: 当前监控指标值。

        Returns:
            首次调用或优于历史最优（含 ``min_delta``）时返回 True。
        """
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, step: int | None = None) -> bool:
        """更新状态机。

        Args:
            value: 当前监控指标值。
            step: 当前轮次（用于记录 best_step）；``None`` 表示按调用次数自增。

        Returns:
            是否应当停止训练。
        """
        current_step = (self.best_step + self.counter + 1) if step is None else step
        if self.is_improvement(value):
            self.best = float(value)
            self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def state_dict(self) -> Mapping[str, Any]:
        """导出状态，供 §20.3 实验记录留痕。"""
        return {
            "best": self.best,
            "best_step": self.best_step,
            "counter": self.counter,
            "should_stop": self.should_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }


@dataclass(slots=True)
class TrainingResult:
    """训练结果摘要（写入实验记录，§20.3）。

    Attributes:
        best_epoch: 监控指标最优的轮次。
        best_metric: 最优监控指标值。
        epochs_run: 实际训练轮数。
        stopped_early: 是否由早停终止。
        history: 每轮的监控指标序列。
        checkpoint_path: 权重输出路径。
    """

    best_epoch: int = 0
    best_metric: float | None = None
    epochs_run: int = 0
    stopped_early: bool = False
    history: Sequence[float] = field(default_factory=tuple)
    checkpoint_path: str = ""


class Trainer:
    """训练循环封装（§10.2、§22）。

    依赖 PyTorch，导入时惰性加载，使本模块在无 torch 环境下仍可被导入与配置校验。
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        """初始化。

        Args:
            config: 训练配置；``None`` 时使用默认值。
            early_stopping: 早停状态机；``None`` 时按 config 构造。
        """
        self.config = config or TrainingConfig()
        self.early_stopping = early_stopping or EarlyStopping(
            patience=self.config.early_stop_patience,
            min_delta=self.config.early_stop_min_delta,
            mode=self.config.monitor_mode,
        )

    def validate(
        self,
        model: Any,
        data_loader: Any,
        loss_fn: Any | None = None,
    ) -> Mapping[str, float]:
        """在给定数据加载器上计算损失与准确率。

        Args:
            model: 模型。
            data_loader: 可迭代的 ``(features, targets)`` 批次。
            loss_fn: 损失函数；``None`` 时使用 ``CrossEntropyLoss``。

        Returns:
            含 ``loss`` 与 ``accuracy`` 的映射。

        Raises:
            ImportError: 未安装 PyTorch。
        """
        torch = _require_torch()
        criterion = loss_fn or torch.nn.CrossEntropyLoss()

        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for features, targets in data_loader:
                logits = model(features)
                loss = criterion(logits, targets)
                batch = int(targets.shape[0])
                total_loss += float(loss.item()) * batch
                total_samples += batch
                correct += int((logits.argmax(dim=1) == targets).sum().item())

        if total_samples == 0:
            return {"loss": float("nan"), "accuracy": 0.0}
        return {
            "loss": total_loss / total_samples,
            "accuracy": correct / total_samples,
        }

    def fit(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any | None = None,
        loss_fn: Any | None = None,
        checkpoint_path: str | None = None,
    ) -> TrainingResult:
        """执行训练循环（§10.2、§22）。

        Args:
            model: 待训练模型。
            train_loader: 训练数据加载器（``(features, targets)`` 批次）。
            val_loader: 验证数据加载器；``None`` 时以训练损失作为监控指标。
            loss_fn: 损失函数；``None`` 时使用 ``CrossEntropyLoss``
                （类别加权见 §10.5，可由调用方传入）。
            checkpoint_path: 权重输出路径；``None`` 表示不落盘。

        Returns:
            TrainingResult（最优轮次、最优指标、实际轮数与是否早停）。

        Raises:
            ImportError: 未安装 PyTorch。
        """
        torch = _require_torch()
        config = self.config

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_weight_decay,
        )
        criterion = loss_fn or torch.nn.CrossEntropyLoss()

        history: list[float] = []
        best_state: dict[str, Any] | None = None

        for epoch in range(1, config.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for features, targets in train_loader:
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                if config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                batch = int(targets.shape[0])
                running += float(loss.item()) * batch
                seen += batch

            train_loss = running / seen if seen else float("nan")
            if val_loader is not None:
                metrics = self.validate(model, val_loader, criterion)
                monitor = float(metrics["loss"]) if config.monitor.endswith("loss") else float(metrics["accuracy"])
            else:
                monitor = train_loss
            history.append(monitor)

            improved = self.early_stopping.is_improvement(monitor)
            if improved:
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stop = self.early_stopping.step(monitor, epoch)
            if stop:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)

        return TrainingResult(
            best_epoch=self.early_stopping.best_step,
            best_metric=self.early_stopping.best,
            epochs_run=len(history),
            stopped_early=bool(history) and len(history) < config.epochs,
            history=tuple(history),
            checkpoint_path=checkpoint_path or "",
        )


def _require_torch() -> Any:
    """惰性导入 PyTorch。

    Raises:
        ImportError: 未安装 torch。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "训练需要 PyTorch，请执行 `pip install -r requirements.txt`"
            "（CPU 版可使用 --index-url https://download.pytorch.org/whl/cpu）"
        ) from exc
    return torch
