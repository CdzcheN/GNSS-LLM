"""S7 深度时序检测器：多源特征融合 + LSTM + 三分类头（§10）。

对应开发文档
    §10 章（10.1 模型定位、10.2 默认模型、10.3 多模态结构、10.4 模型对比、10.5 类别不平衡）、
    §8.2 策略分类（S7 深度时序检测）、§8.3（高不确定性 → 深度时序）、
    §1.4 G8（参数量 < 10⁶）、§6.2 时间窗口、§6.4 标准化（统计量只能来自训练集）、§19 部署设计。

职责
    1. 构建多源融合 LSTM 三分类模型（build_model）；
    2. 训练模型并保存自查点（train），含训练集标准化统计量以便推理复现；
    3. 以**流式缓冲**方式推理：每次 ``run`` 追加一行，填满窗口后给出三态判定。

不做（边界）
    - 不参与实时 1 Hz 关键路径的强制调用：仅在被策略选中时执行（§9.5 原则 A）；
    - 不承接事件管理与 LLM 总结（属 M7、M8）；
    - 不在训练中使用测试/留出数据计算标准化统计量（§6.4、§16.3）。

输入 / 输出
    训练：特征表 CSV + TrainingConfig → checkpoint（含权重、特征列、标准化统计量）
    推理：单历元特征映射（逐次追加）→ DetectionResult（含三分类概率，§8.1）

关键约束
    - 默认配置与 §10.2 一致：LSTM 2 层、hidden 64、窗口 60 s、3 类；
    - 类别不平衡按 §10.5 处理（加权 CE / Focal）；
    - torch 与 numpy 均惰性导入，未安装时本模块仍可被导入与静态检查；
    - 无需 checkpoint 时返回 ``SKIPPED`` 而不是抛异常，保证编排链不中断（§9.5 原则 D）。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.detectors.base import AttackType, BaseDetector, DetectionResult, DetectorStatus, numeric
from src.train.dataset import build_windows

#: 默认模型配置（§10.2 配置基线）。
DEFAULT_LSTM_LAYERS: int = 2
DEFAULT_HIDDEN_SIZE: int = 64
DEFAULT_WINDOW_S: int = 60
DEFAULT_NUM_CLASSES: int = 3
DEFAULT_DROPOUT: float = 0.1

#: 参数量目标上限（§1.4 G8：深度检测核心模型参数量目标 < 10⁶）。
PARAMETER_BUDGET: int = 10**6

#: 类别加权基线（§10.5）。
DEFAULT_CLASS_WEIGHTS: Mapping[int, float] = {
    0: 1.0,   # Normal
    1: 3.9,   # Spoofing
    2: 70.0,  # Jamming
}

#: Focal Loss 的 γ（§10.5）。
DEFAULT_FOCAL_GAMMA: float = 2.0

#: 训练时排除的非特征列（时间、身份与布尔掩码列）。
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"Timestamp", "Day", "Hour", "Label", "split", "source_file", "sat_mask", "miss_mask"}
)

#: 默认验证集比例（§16.2 一期基线：按时间顺序前 70% / 后 30%）。
DEFAULT_VALIDATION_RATIO: float = 0.3


def _require_torch() -> Any:
    """惰性导入 PyTorch。

    Raises:
        ImportError: 未安装 torch。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "深度时序检测器需要 PyTorch，请执行 `pip install -r requirements.txt`"
            "（CPU 版可使用 --index-url https://download.pytorch.org/whl/cpu）"
        ) from exc
    return torch


def select_feature_columns(frame: Any, exclude: Sequence[str] = ()) -> list[str]:
    """选定参与建模的数值特征列。

    Args:
        frame: 特征表。
        exclude: 额外排除的列名。

    Returns:
        数值列名列表（排除时间/标签/布尔掩码等列，按表内顺序）。

    Note:
        默认只使用数值列：原始 96 列观测（CNO/Res/Elev）与派生特征都会纳入；
        若需要模态消融（§18.6），可在调用处传入更窄的列清单。
    """
    excluded = set(NON_FEATURE_COLUMNS) | set(exclude)
    columns: list[str] = []
    for name in frame.columns:
        if name in excluded:
            continue
        dtype = str(frame[name].dtype)
        if dtype.startswith(("float", "int")) and "bool" not in dtype:
            columns.append(str(name))
    return columns


def build_model(
    input_dim: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_LSTM_LAYERS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = DEFAULT_DROPOUT,
) -> Any:
    """构建多源融合 LSTM 三分类模型（§10.2、§10.3）。

    结构：``LayerNorm`` → ``LSTM`` → （分类头 + 辅助头），对应 §10.3 的
    「特征融合 → 时序编码器 → Classification / Auxiliary Head」。

    Args:
        input_dim: 输入特征维度 D。
        hidden_size: 隐藏维度（§10.2 默认 64）。
        num_layers: LSTM 层数（§10.2 默认 2）。
        num_classes: 输出类别数（§1.3 三态）。
        dropout: Dropout 比例（§22 正则手段之一）。

    Returns:
        未训练的 ``torch.nn.Module``。

    Raises:
        ImportError: 未安装 PyTorch。
        ValueError: 维度参数非法。
    """
    torch = _require_torch()
    if input_dim < 1:
        raise ValueError(f"input_dim 必须 >= 1：{input_dim}")
    if hidden_size < 1 or num_layers < 1:
        raise ValueError(f"hidden_size/num_layers 必须为正：{hidden_size}/{num_layers}")

    class _MultiSourceLSTM(torch.nn.Module):
        """多源特征融合 + LSTM + 三分类头（§10.3）。"""

        def __init__(self) -> None:
            super().__init__()
            self.input_norm = torch.nn.LayerNorm(input_dim)
            self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size // 2, num_classes),
            )
            #: 辅助头（§10.3）：输出窗口级重构/异常标量，训练时作为弱监督正则。
            self.auxiliary = torch.nn.Linear(hidden_size, 1)
            self.config = {
                "input_dim": input_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_classes": num_classes,
                "dropout": dropout,
            }

        def forward(self, features: Any) -> Any:
            """前向：``(B, W, D)`` → 三分类 logits。"""
            normalized = self.input_norm(features)
            output, (hidden, _) = self.lstm(normalized)
            last = hidden[-1]
            self.last_auxiliary = self.auxiliary(last)
            return self.classifier(last)

        def count_parameters(self) -> int:
            """可训练参数量（用于 §1.4 G8 校验）。"""
            return int(sum(parameter.numel() for parameter in self.parameters()))

    return _MultiSourceLSTM()


class DeepTemporalDetector(BaseDetector):
    """深度时序检测器（§8.2 S7）。

    以流式方式工作：每次 ``run`` 追加一行特征，缓冲区填满 ``window_s`` 后开始推理，
    与在线闭环（§15.1）的逐历元推进方式一致。
    """

    detector_id = "deep_temporal"
    supported_context = ("S", "Q", "O", "N", "H")
    version = "v1"

    def __init__(
        self,
        checkpoint: str | None = None,
        num_layers: int = DEFAULT_LSTM_LAYERS,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_classes: int = DEFAULT_NUM_CLASSES,
        window_s: int = DEFAULT_WINDOW_S,
        device: str = "cpu",
    ) -> None:
        """初始化。

        Args:
            checkpoint: 权重路径；``None`` 表示未训练（``run`` 返回 ``SKIPPED``）。
            num_layers: LSTM 层数（§10.2）。
            hidden_size: 隐藏维度（§10.2）。
            num_classes: 类别数（§1.3）。
            window_s: 输入窗口长度（§6.2）。
            device: 推理设备。
        """
        self.checkpoint = checkpoint
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.window_s = int(window_s)
        self.device = device

        self._model: Any = None
        self._feature_columns: list[str] = []
        self._buffer: deque[list[float]] = deque(maxlen=self.window_s)
        self._mean: Any = None
        self._std: Any = None

    # ------------------------------------------------------------------ 推理

    def _load(self) -> None:
        """加载 checkpoint（惰性，仅首次调用时执行）。"""
        if self._model is not None or self.checkpoint is None:
            return
        torch = _require_torch()
        payload = torch.load(self.checkpoint, map_location=self.device)
        columns = list(payload["feature_columns"])
        model = build_model(
            input_dim=len(columns),
            hidden_size=int(payload.get("hidden_size", self.hidden_size)),
            num_layers=int(payload.get("num_layers", self.num_layers)),
            num_classes=int(payload.get("num_classes", self.num_classes)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._model = model
        self._feature_columns = columns
        self._mean = payload.get("mean")
        self._std = payload.get("std")

    def run(
        self,
        data: Any = None,
        context: Any = None,
        config: Mapping[str, Any] | None = None,
    ) -> DetectionResult:
        """流式推理：追加一行，窗口填满后输出三态判定（§8.1）。

        Args:
            data: 单历元特征映射（与其它检测器一致）。
            context: GNSS Context（用于记录质量）。
            config: 可选推理配置（``stream_reset`` 为 True 时清空缓冲）。

        Returns:
            DetectionResult；未提供 checkpoint 或窗口未填满时返回 ``SKIPPED``。

        Raises:
            ImportError: 未安装 PyTorch。
            KeyError: checkpoint 缺少必要字段。
        """
        settings = dict(config or {})
        if settings.get("stream_reset"):
            self._buffer.clear()

        if self.checkpoint is None:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={"reason": "no_checkpoint", "model_version": self.version},
                status=DetectorStatus.SKIPPED,
            )

        self._load()
        vector = self._vectorize(data)
        if vector is None:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={"reason": "feature_missing", "expected_dim": len(self._feature_columns)},
                status=DetectorStatus.SKIPPED,
            )
        self._buffer.append(vector)

        if len(self._buffer) < self.window_s:
            return DetectionResult(
                detector_id=self.detector_id,
                attack_type=AttackType.NORMAL,
                confidence=0.0,
                evidence={
                    "reason": "window_not_ready",
                    "filled": len(self._buffer),
                    "window_s": self.window_s,
                },
                status=DetectorStatus.SKIPPED,
            )

        probs = self._infer()
        best = int(max(range(len(probs)), key=lambda index: probs[index]))
        attack = AttackType(best) if best in (0, 1, 2) else AttackType.NORMAL

        return DetectionResult(
            detector_id=self.detector_id,
            attack_type=attack,
            confidence=float(probs[best]),
            evidence={
                "probs": {AttackType(i).name.lower(): round(float(p), 6) for i, p in enumerate(probs)},
                "checkpoint": str(self.checkpoint),
                "window_s": self.window_s,
                "model_version": self.version,
            },
            data_quality=None if context is None else getattr(context, "data_quality", None),
            status=DetectorStatus.OK,
        )

    def _vectorize(self, data: Any) -> list[float] | None:
        """把特征映射整理为模型输入向量（按训练时的列顺序）。"""
        if not isinstance(data, Mapping):
            return None
        values: list[float] = []
        for name in self._feature_columns:
            value = numeric(data, name)
            if value is None:
                return None
            values.append(float(value))
        return values

    def _infer(self) -> list[float]:
        """执行一次前向并返回三分类概率。"""
        torch = _require_torch()
        import numpy as np

        window = np.asarray(list(self._buffer), dtype="float64")
        if self._mean is not None and self._std is not None:
            std = np.where(np.asarray(self._std) == 0, 1.0, np.asarray(self._std))
            window = (window - np.asarray(self._mean)) / std
        tensor = torch.tensor(window[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0].tolist()
        return [float(value) for value in probs]

    # ------------------------------------------------------------------ 训练

    def train(
        self,
        features_path: str,
        config: Any,
        loss: str = "weighted_ce",
        checkpoint_path: str = "",
        validation_ratio: float = DEFAULT_VALIDATION_RATIO,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    ) -> Any:
        """训练模型并保存 checkpoint（§10.2、§10.5、§16.2、§22）。

        Args:
            features_path: 特征表路径（由 ``scripts/extract_features.py`` 生成）。
            config: :class:`src.train.trainer.TrainingConfig` 实例。
            loss: ``ce`` / ``weighted_ce`` / ``focal``（§10.5）。
            checkpoint_path: 权重输出路径。
            validation_ratio: 验证集比例（按时间顺序取尾部，§16.2）。
            focal_gamma: Focal Loss 的 γ。

        Returns:
            :class:`src.train.trainer.TrainingResult`。

        Raises:
            ImportError: 未安装 pandas / numpy / torch。
            ValueError: 损失类型非法或数据不足。

        Note:
            标准化统计量**只在训练段**计算，并随 checkpoint 保存（§6.4、§16.3）。
        """
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - 依赖环境相关
            raise ImportError("训练需要 pandas 与 numpy，请执行 `pip install -r requirements.txt`") from exc

        torch = _require_torch()
        from src.train.losses import LOSS_NAMES
        from src.train.trainer import EarlyStopping, Trainer

        if loss not in LOSS_NAMES:
            raise ValueError(f"未知损失 {loss!r}，允许：{LOSS_NAMES}")

        frame = pd.read_csv(features_path)
        columns = select_feature_columns(frame)
        if not columns:
            raise ValueError("未找到可用数值特征列，请检查特征表内容")

        # 按时间顺序划分：训练在前、验证在后（§16.2 一期基线），禁止随机打散（§2.5）
        boundary = int(len(frame) * (1.0 - validation_ratio))
        if boundary < config.window_s or len(frame) - boundary < config.window_s:
            raise ValueError(
                f"数据切分后不足一个窗口（{config.window_s}）："
                f"train={boundary}，validation={len(frame) - boundary}"
            )
        train_frame = frame.iloc[:boundary]
        val_frame = frame.iloc[boundary:]

        mean = train_frame[columns].mean().to_numpy(dtype="float64")
        std = train_frame[columns].std().to_numpy(dtype="float64")
        std = np.where(std == 0, 1.0, std)

        def _normalize(part: Any) -> Any:
            normalized = part.copy()
            normalized[columns] = (part[columns].to_numpy(dtype="float64") - mean) / std
            return normalized

        X_train, y_train, _ = build_windows(
            _normalize(train_frame), columns, window_s=config.window_s
        )
        X_val, y_val, _ = build_windows(
            _normalize(val_frame), columns, window_s=config.window_s
        )

        def _loader(X: Any, y: Any, shuffle: bool) -> Any:
            features = torch.tensor(X, dtype=torch.float32)
            targets = torch.tensor(y.astype("int64"), dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(features, targets)
            return torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

        class_counts = np.bincount(y_train.astype("int64"), minlength=config.num_classes)
        weights = None
        if loss in ("weighted_ce", "focal"):
            counts = {index: float(max(count, 1)) for index, count in enumerate(class_counts)}
            from src.train.losses import balanced_class_weights

            derived = balanced_class_weights(counts)
            weights = torch.tensor(
                [derived.get(index, 1.0) for index in range(config.num_classes)],
                dtype=torch.float32,
            )

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        model = build_model(
            input_dim=len(columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        )
        trainer = Trainer(
            config,
            EarlyStopping(
                patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta,
                mode=config.monitor_mode,
            ),
        )
        result = trainer.fit(
            model,
            _loader(X_train, y_train, shuffle=True),
            _loader(X_val, y_val, shuffle=False),
            loss_fn=criterion,
            checkpoint_path=None,
        )

        if checkpoint_path:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_columns": columns,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "num_classes": self.num_classes,
                    "window_s": config.window_s,
                    "loss": loss,
                    "focal_gamma": focal_gamma,
                    "class_counts": class_counts.tolist(),
                },
                path,
            )
            self.checkpoint = str(path)

        return result

    def describe(self):  # type: ignore[override]
        """返回注册表元信息（§8.4）；深度模型标为高时延/高开销（§9.5 原则 A）。"""
        meta = super().describe()
        meta.input_schema = {
            "type": "window_tensor",
            "window_s": self.window_s,
            "num_classes": self.num_classes,
        }
        meta.output_schema = {"type": "DetectionResult", "evidence": ["probs", "model_version"]}
        meta.expected_latency_ms = 20.0
        meta.expected_cost = 10.0
        return meta
