"""评估结果可视化：把指标与预测画成可读的图（§17、§18、§20.3）。

对应开发文档
    §17 评价指标体系（逐秒 / 类别 / 事件层）、§18 消融与对比实验、
    §20.3 实验记录规范（``figure_paths`` 字段）、§6.3 派生特征（特征曲线）、
    §7.3 上下文特征、§13.1 事件时间范围。

职责
    1. **预测 vs 真实的时间序列对比**：最直观地看出“检测在哪些时段出错”；
    2. 混淆矩阵热图与各类别 P/R/F1 柱状图（§17.1、§17.2，Jamming 单列）；
    3. **关键特征曲线**，并叠加真实异常区间底色，用于判断“物理上是否可分”；
    4. 事件时间线（真实事件 vs 检测事件，§17.3）；
    5. 消融 / 对比实验柱状图（§18）。

不做（边界）
    - 不计算指标（由 ``src/eval/metrics.py`` 负责），只做绘制；
    - 不修改输入数据（只读，产出图像文件）；
    - 不联网、不加载外部字体文件，只使用系统已安装字体。

输入 / 输出
    输入：含 ``Label`` / ``prediction`` 列的预测表、特征表、事件区间
    输出：PNG 图像文件路径（可直接填入 §20.3 的 ``figure_paths``）

关键约束
    - 使用 **Agg 无界面后端**，保证在服务器 / CI 上也能出图；
    - 中文字体“尽力而为”：有 CJK 字体则用中文标注，否则自动退回英文，**不因字体缺失而失败**；
    - 每个绘图函数都返回**已写出的文件路径**，便于记录到实验日志；
    - 超长序列按等间隔抽样（保持时间顺序），避免图件过大与渲染过慢。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

#: 图像 DPI 与默认尺寸。
DEFAULT_DPI: int = 120
#: 单张图最多绘制的点数（超出则等间隔抽样，保持时间顺序）。
DEFAULT_MAX_POINTS: int = 20000

#: 中文字体优先级（找不到时退回英文标注）。
CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK HK",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
)

#: 三态标签的显示名（§1.3）。
CLASS_DISPLAY: Mapping[int, str] = {0: "Normal", 1: "Spoofing", 2: "Jamming"}


def _prepare() -> tuple[Any, bool]:
    """导入 matplotlib（Agg 后端）并选定字体。

    Returns:
        ``(pyplot 模块, 是否可用中文)``。

    Raises:
        ImportError: 未安装 matplotlib。
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError(
            "可视化需要 matplotlib，请执行 `pip install -r requirements.txt`"
        ) from exc

    available = {font.name for font in font_manager.fontManager.ttflist}
    picked = next((name for name in CHINESE_FONT_CANDIDATES if name in available), None)
    if picked:
        matplotlib.rcParams["font.family"] = picked
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.autolayout"] = True
    return plt, picked is not None


def _t(use_chinese: bool, zh: str, en: str) -> str:
    """按字体可用性选择标签文本。

    Args:
        use_chinese: 是否可用中文。
        zh: 中文文本。
        en: 英文文本。

    Returns:
        实际使用的文本。
    """
    return zh if use_chinese else en


def _require_pandas() -> Any:
    """惰性导入 pandas。"""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError("可视化需要 pandas，请执行 `pip install -r requirements.txt`") from exc
    return pd


def _sample(frame: Any, max_points: int) -> Any:
    """对超长序列等间隔抽样（保持时间顺序）。

    Args:
        frame: 输入表。
        max_points: 最大点数（<=0 表示不抽样）。

    Returns:
        抽样后的表。
    """
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    step = len(frame) // max_points + 1
    return frame.iloc[::step]


def _save(fig: Any, path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """保存图像并释放资源。

    Args:
        fig: matplotlib Figure。
        path: 输出路径。
        dpi: 分辨率。

    Returns:
        写出的文件路径。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return target


def plot_prediction_timeline(
    frame: Any,
    out_path: str | Path,
    true_column: str = "Label",
    pred_column: str = "prediction",
    max_points: int = DEFAULT_MAX_POINTS,
    title: str | None = None,
) -> Path:
    """绘制“真实 vs 预测 vs 不一致”的时间序列三联图（§17.1）。

    这是判断“检测哪里出错”最直接的图：第三条带显示预测与真实不一致的时段。

    Args:
        frame: 含真实列与预测列的预测结果表（需保持时间顺序）。
        out_path: 输出图像路径。
        true_column: 真实标签列。
        pred_column: 预测标签列。
        max_points: 最大点数（超出则抽样）。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 缺少必需列。
    """
    plt, zh = _prepare()
    import numpy as np

    for column in (true_column, pred_column):
        if column not in frame.columns:
            raise ValueError(f"缺少列 {column!r}（现有列：{list(frame.columns)[:8]}…）")

    data = _sample(frame, max_points)
    truth = data[true_column].to_numpy()
    prediction = data[pred_column].to_numpy()
    x = np.arange(len(data))
    mismatch = truth != prediction

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    axes[0].step(x, truth, where="post", color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel(_t(zh, "真实标签", "Ground truth"), fontsize=10)
    axes[0].set_yticks(list(CLASS_DISPLAY))
    axes[0].set_yticklabels([CLASS_DISPLAY[k] for k in CLASS_DISPLAY], fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].step(x, prediction, where="post", color="tab:red", linewidth=1.0)
    axes[1].set_ylabel(_t(zh, "预测标签", "Prediction"), fontsize=10)
    axes[1].set_yticks(list(CLASS_DISPLAY))
    axes[1].set_yticklabels([CLASS_DISPLAY[k] for k in CLASS_DISPLAY], fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(x, 0, 1, where=mismatch, color="tab:orange", alpha=0.7, step="post")
    axes[2].set_ylabel(_t(zh, "预测错误", "Mismatch"), fontsize=10)
    axes[2].set_yticks([])
    axes[2].set_xlabel(_t(zh, "历元序号", "Epoch index"), fontsize=10)
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    mismatch_ratio = float(mismatch.mean()) if len(mismatch) else 0.0
    fig.suptitle(
        title
        or _t(
            zh,
            f"预测与真实对比（不一致率 {mismatch_ratio:.3f}）",
            f"Prediction vs truth (mismatch rate {mismatch_ratio:.3f})",
        ),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_confusion_matrix(
    matrix: Mapping[int, Mapping[int, int]],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """绘制混淆矩阵热图（§17.1）。

    Args:
        matrix: ``{真实标签: {预测标签: 计数}}``（来自 ``metrics.confusion_matrix``）。
        out_path: 输出图像路径。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
    """
    plt, zh = _prepare()
    import numpy as np

    labels = sorted(matrix)
    values = np.array([[matrix[true][pred] for pred in labels] for true in labels], dtype="float64")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    tick_labels = [CLASS_DISPLAY.get(label, str(label)) for label in labels]
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel(_t(zh, "预测", "Predicted"), fontsize=10)
    ax.set_ylabel(_t(zh, "真实", "Actual"), fontsize=10)

    # 单元格内标注计数（行归一化比例一并给出，便于识别少数类）
    row_sums = values.sum(axis=1, keepdims=True)
    for row in range(len(labels)):
        for col in range(len(labels)):
            count = int(values[row, col])
            ratio = values[row, col] / row_sums[row, 0] if row_sums[row, 0] else 0.0
            color = "white" if values[row, col] > values.max() * 0.6 else "black"
            ax.text(col, row, f"{count}\n{ratio:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title(title or _t(zh, "混淆矩阵（计数 / 行占比）", "Confusion matrix (count / row ratio)"),
                 fontsize=12)
    return _save(fig, out_path)


def plot_class_metrics(
    report: Mapping[str, Any],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """绘制各类别 Precision / Recall / F1 柱状图（§17.2，Jamming 单列）。

    Args:
        report: ``metrics.classification_report`` 的输出（含 ``per_class``）。
        out_path: 输出图像路径。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 报告缺少 ``per_class``。
    """
    plt, zh = _prepare()
    import numpy as np

    per_class = report.get("per_class")
    if not per_class:
        raise ValueError("报告缺少 per_class，请先调用 metrics.classification_report")

    names = list(per_class)
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, metric in enumerate(metrics):
        values = [float(per_class[name].get(metric, 0.0)) for name in names]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=metric.capitalize())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in names], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(_t(zh, "指标值", "Score"), fontsize=10)
    ax.axhline(float(report.get("macro_f1", 0.0)), color="gray", linestyle="--", linewidth=1,
               label=f"Macro-F1 = {float(report.get('macro_f1', 0.0)):.3f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        title
        or _t(zh, "各类别指标（Jamming 单列，§17.2）", "Per-class metrics (Jamming reported separately)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_feature_timeline(
    frame: Any,
    columns: Sequence[str],
    out_path: str | Path,
    label_column: str = "Label",
    max_points: int = DEFAULT_MAX_POINTS,
    title: str | None = None,
) -> Path:
    """绘制关键特征曲线，并用底色标出真实异常区间（§6.3、§7.3）。

    用于判断“异常在物理特征上是否可分”——若异常区间内特征没有明显变化，
    说明该特征对当前攻击类型无区分度，问题不在检测器阈值上。

    Args:
        frame: 特征表（含标签列）。
        columns: 要绘制的特征列（每个一子图）。
        out_path: 输出图像路径。
        label_column: 标签列（用于标注异常区间）。
        max_points: 最大点数（抽样）。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 特征列为空或列不存在。
    """
    plt, zh = _prepare()
    import numpy as np

    if not columns:
        raise ValueError("columns 不能为空")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少特征列：{missing}")

    data = _sample(frame, max_points)
    x = np.arange(len(data))
    labels = data[label_column].to_numpy() if label_column in data.columns else np.zeros(len(data))
    abnormal = labels != 0

    fig, axes = plt.subplots(len(columns), 1, figsize=(15, 2.6 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, name in zip(axes, columns):
        ax.plot(x, data[name].to_numpy(dtype="float64"), linewidth=1.0, color="tab:green")
        ax.fill_between(x, *ax.get_ylim(), where=abnormal, color="tab:red", alpha=0.12, step="post")
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel(_t(zh, "历元序号", "Epoch index"), fontsize=10)
    axes[0].set_title(
        title
        or _t(
            zh,
            "关键特征曲线（红色底色 = 真实异常区间）",
            "Key features (red shading = ground-truth anomaly)",
        ),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_event_timeline(
    truth: Sequence[Any],
    predicted: Sequence[Any],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """绘制事件时间线：真实事件与检测事件的区间对比（§17.3）。

    Args:
        truth: 真实事件区间（``metrics.Interval``）。
        predicted: 检测事件区间。
        out_path: 输出图像路径。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib。
    """
    plt, zh = _prepare()

    fig, ax = plt.subplots(figsize=(13, 4))
    for index, interval in enumerate(truth):
        ax.barh(1.0, interval.end - interval.start, left=interval.start, height=0.35,
                color="tab:blue", alpha=0.85, label="Ground truth" if index == 0 else None)
    for index, interval in enumerate(predicted):
        ax.barh(0.5, interval.end - interval.start, left=interval.start, height=0.35,
                color="tab:red", alpha=0.85, label="Detected" if index == 0 else None)

    ax.set_yticks([1.0, 0.5])
    ax.set_yticklabels([_t(zh, "真实事件", "Ground truth"), _t(zh, "检测事件", "Detected")], fontsize=9)
    ax.set_xlabel(_t(zh, "时间（秒）", "Time (s)"), fontsize=10)
    ax.set_ylim(0.2, 1.3)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        title
        or _t(
            zh,
            f"事件时间线（真实 {len(truth)} / 检测 {len(predicted)}）",
            f"Event timeline (truth {len(truth)} / detected {len(predicted)})",
        ),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_ablation_comparison(
    records: Mapping[str, float],
    out_path: str | Path,
    metric_name: str = "macro_f1",
    title: str | None = None,
) -> Path:
    """绘制消融 / 对比实验柱状图（§18）。

    Args:
        records: 对照组名 → 指标值。
        out_path: 输出图像路径。
        metric_name: 指标名（用于轴标签）。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 输入为空。
    """
    plt, zh = _prepare()
    import numpy as np

    if not records:
        raise ValueError("records 不能为空")

    names = list(records)
    values = [float(records[name]) for name in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 4.5))
    bars = ax.bar(x, values, color="tab:purple", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_ylim(0, max(values + [0.0]) * 1.2 + 0.02)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title or _t(zh, f"消融对比：{metric_name}", f"Ablation comparison: {metric_name}"),
                 fontsize=12)
    return _save(fig, out_path)


def plot_label_distribution(
    frame: Any,
    out_path: str | Path,
    true_column: str = "Label",
    pred_column: str = "prediction",
    title: str | None = None,
) -> Path:
    """绘制真实与预测的标签分布对比（§17.2）。

    一眼看出“模型是否只会输出某一类”（例如全判 Normal）。

    Args:
        frame: 预测结果表。
        out_path: 输出图像路径。
        true_column: 真实标签列。
        pred_column: 预测标签列。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 缺少列。
    """
    plt, zh = _prepare()
    import numpy as np

    for column in (true_column, pred_column):
        if column not in frame.columns:
            raise ValueError(f"缺少列 {column!r}")

    labels = sorted(CLASS_DISPLAY)
    truth_counts = [int((frame[true_column] == label).sum()) for label in labels]
    pred_counts = [int((frame[pred_column] == label).sum()) for label in labels]
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(x - width / 2, truth_counts, width, label=_t(zh, "真实", "Truth"), color="tab:blue")
    axes[0].bar(x + width / 2, pred_counts, width, label=_t(zh, "预测", "Prediction"), color="tab:red")
    for offset, counts in ((-width / 2, truth_counts), (width / 2, pred_counts)):
        for position, value in zip(x, counts):
            axes[0].text(position + offset, value, str(value), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CLASS_DISPLAY[label] for label in labels], fontsize=9)
    axes[0].set_ylabel(_t(zh, "样本数", "Count"), fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    total_true = sum(truth_counts) or 1
    total_pred = sum(pred_counts) or 1
    axes[1].bar(x - width / 2, [c / total_true for c in truth_counts], width,
                label=_t(zh, "真实", "Truth"), color="tab:blue", alpha=0.85)
    axes[1].bar(x + width / 2, [c / total_pred for c in pred_counts], width,
                label=_t(zh, "预测", "Prediction"), color="tab:red", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CLASS_DISPLAY[label] for label in labels], fontsize=9)
    axes[1].set_ylabel(_t(zh, "占比", "Ratio"), fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(title or _t(zh, "标签分布：真实 vs 预测", "Label distribution: truth vs prediction"),
                 fontsize=12)
    return _save(fig, out_path)


def plot_confidence_distribution(
    frame: Any,
    out_path: str | Path,
    true_column: str = "Label",
    score_column: str = "confidence",
    title: str | None = None,
) -> Path:
    """按真实标签分组绘制置信度分布（§17.1）。

    若各类别的置信度分布高度重叠，说明**分数没有区分度**——此时无论怎么调阈值都难以分开，
    应回到特征层（§6.3）而不是继续调门限。

    Args:
        frame: 预测结果表（需含分数列）。
        out_path: 输出图像路径。
        true_column: 真实标签列。
        score_column: 置信度列。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 缺少列。
    """
    plt, zh = _prepare()
    import numpy as np

    for column in (true_column, score_column):
        if column not in frame.columns:
            raise ValueError(f"缺少列 {column!r}")

    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label in sorted(CLASS_DISPLAY):
        values = frame.loc[frame[true_column] == label, score_column].to_numpy(dtype="float64")
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, alpha=0.55, label=f"{CLASS_DISPLAY[label]} (n={values.size})")
        ax.axvline(float(np.mean(values)), linestyle="--", linewidth=1)

    ax.set_xlabel(_t(zh, "置信度", "Confidence"), fontsize=10)
    ax.set_ylabel(_t(zh, "样本数", "Count"), fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(
        title
        or _t(zh, "按真实标签分组的置信度分布（虚线为组均值）",
              "Confidence by ground-truth class (dashed = class mean)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_feature_distributions(
    frame: Any,
    columns: Sequence[str],
    out_path: str | Path,
    label_column: str = "Label",
    title: str | None = None,
) -> Path:
    """绘制“正常 vs 异常”区间的特征箱线图（§6.3、§7.3）。

    这是判断**特征可分性**最直接的图：若异常组与正常组的箱体几乎重合，
    该特征对当前攻击无区分度。

    Args:
        frame: 特征表（含标签列）。
        columns: 特征列（每个一子图）。
        out_path: 输出图像路径。
        label_column: 标签列。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 列缺失或为空。
    """
    plt, zh = _prepare()
    import numpy as np

    if not columns:
        raise ValueError("columns 不能为空")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少特征列：{missing}")
    if label_column not in frame.columns:
        raise ValueError(f"缺少标签列 {label_column!r}")

    columns_per_row = 4
    rows = int(np.ceil(len(columns) / columns_per_row))
    fig, axes = plt.subplots(rows, columns_per_row, figsize=(4.2 * columns_per_row, 3.2 * rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, name in zip(flat, columns):
        groups = [
            frame.loc[frame[label_column] == label, name].dropna().to_numpy(dtype="float64")
            for label in sorted(CLASS_DISPLAY)
        ]
        ax.boxplot(groups, tick_labels=[CLASS_DISPLAY[label] for label in sorted(CLASS_DISPLAY)],
                   showfliers=False)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in flat[len(columns):]:
        ax.axis("off")

    fig.suptitle(
        title or _t(zh, "特征分布：正常 vs 异常（箱体越分离可区分性越好）",
                    "Feature distributions by class (more separated = better)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_feature_discriminability(
    frame: Any,
    columns: Sequence[str],
    out_path: str | Path,
    label_column: str = "Label",
    title: str | None = None,
) -> Path:
    """按区分度排序绘制单特征条形图（§6.3）。

    区分度采用标准化均值差（绝对值）：

    .. math:: d = |mean_{异常} - mean_{正常}| / (std_{异常} + std_{正常})

    值越大越能分开两类；接近 0 表示该特征对本数据集几乎无信息量。

    Args:
        frame: 特征表（含标签列）。
        columns: 参与评估的特征列。
        out_path: 输出图像路径。
        label_column: 标签列。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 列缺失或为空。
    """
    plt, zh = _prepare()
    import numpy as np

    scores = feature_discriminability(frame, columns, label_column)
    if not scores:
        raise ValueError("没有可用于评估的特征列")

    names = list(scores)
    values = [scores[name] for name in names]
    order = np.argsort(values)
    names = [names[index] for index in order]
    values = [values[index] for index in order]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(names))))
    bars = ax.barh(np.arange(len(names)), values, color="tab:cyan", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(_t(zh, "区分度（标准化均值差）", "Discriminability (standardized mean diff)"),
                  fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        title or _t(zh, "单特征区分度排序（越靠右越有用）",
                    "Per-feature discriminability (right = more useful)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def feature_discriminability(
    frame: Any,
    columns: Sequence[str],
    label_column: str = "Label",
) -> dict[str, float]:
    """计算各特征的区分度（标准化均值差），供排序图与诊断使用。

    Args:
        frame: 特征表（含标签列）。
        columns: 特征列。
        label_column: 标签列。

    Returns:
        特征名 → 区分度（0 表示无区分度，越大越可分）；无法计算时该特征被跳过。

    Raises:
        ValueError: 标签列缺失。
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 依赖环境相关
        raise ImportError("需要 numpy，请执行 `pip install -r requirements.txt`") from exc

    if label_column not in frame.columns:
        raise ValueError(f"缺少标签列 {label_column!r}")

    normal = frame.loc[frame[label_column] == 0]
    abnormal = frame.loc[frame[label_column] != 0]
    scores: dict[str, float] = {}
    for name in columns:
        if name not in frame.columns:
            continue
        left = normal[name].dropna().to_numpy(dtype="float64")
        right = abnormal[name].dropna().to_numpy(dtype="float64")
        if left.size == 0 or right.size == 0:
            continue
        spread = float(np.std(left) + np.std(right))
        if spread <= 0:
            continue
        scores[name] = abs(float(np.mean(right) - np.mean(left))) / spread
    return scores


def plot_roc_pr_curves(
    frame: Any,
    out_path: str | Path,
    score_column: str = "confidence",
    true_column: str = "Label",
    title: str | None = None,
) -> Path:
    """绘制二分类 ROC 与 PR 曲线（约定：Normal vs 非 Normal）。

    对应 §17.1 的 AUC-ROC / AUC-PR。**不依赖 scikit-learn**：直接对分数排序后累积计算。

    Args:
        frame: 预测结果表（含分数列）。
        out_path: 输出图像路径。
        score_column: 分数列（此处用置信度作为异常分数）。
        true_column: 真实标签列。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 缺少列或只有一个类别。
    """
    plt, zh = _prepare()
    import numpy as np

    for column in (true_column, score_column):
        if column not in frame.columns:
            raise ValueError(f"缺少列 {column!r}")

    scores = frame[score_column].to_numpy(dtype="float64")
    positive = (frame[true_column].to_numpy() != 0).astype("int64")
    if positive.min() == positive.max():
        raise ValueError("真实标签只有一个类别，无法计算 ROC / PR（请换用含异常的划分）")

    fpr, tpr, precision, recall, auc_roc, auc_pr = _binary_curves(scores, positive)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="tab:blue", linewidth=1.5, label=f"AUC = {auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("FPR", fontsize=10)
    axes[0].set_ylabel("TPR", fontsize=10)
    axes[0].set_title("ROC", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, color="tab:orange", linewidth=1.5, label=f"AP = {auc_pr:.3f}")
    axes[1].set_xlabel(_t(zh, "召回率", "Recall"), fontsize=10)
    axes[1].set_ylabel(_t(zh, "精确率", "Precision"), fontsize=10)
    axes[1].set_title("Precision-Recall", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        title
        or _t(zh, "以置信度为异常分数的 ROC / PR（Normal vs 非 Normal）",
              "ROC / PR using confidence as anomaly score (Normal vs non-Normal)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def _binary_curves(scores: Any, positive: Any) -> tuple[Any, ...]:
    """由分数与二值标签计算 ROC / PR 曲线与 AUC。

    Args:
        scores: 异常分数（越大越可能为异常）。
        positive: 二值标签（1 = 异常）。

    Returns:
        ``(fpr, tpr, precision, recall, auc_roc, auc_pr)``。

    Note:
        采用“按分数降序逐个作为阈值”的标准做法，用梯形法积分；
        不使用未来信息，也不做随机下采样，保证可复现。
    """
    import numpy as np

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    labels = positive[order]

    total_positive = int(labels.sum())
    total_negative = int(len(labels) - total_positive)

    # 只在分数变化处取阈值点
    distinct = np.where(np.diff(ordered_scores))[0]
    thresholds = np.r_[distinct, len(ordered_scores) - 1]

    true_positive = np.cumsum(labels)[thresholds].astype("float64")
    predicted_positive = (thresholds + 1).astype("float64")
    false_positive = predicted_positive - true_positive

    tpr = true_positive / total_positive if total_positive else np.zeros_like(true_positive)
    fpr = false_positive / total_negative if total_negative else np.zeros_like(false_positive)
    precision = np.divide(true_positive, predicted_positive,
                          out=np.ones_like(true_positive), where=predicted_positive > 0)
    recall = tpr

    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]

    # numpy 2.x 将 trapz 更名为 trapezoid，这里做兼容
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc_roc = float(trapezoid(tpr, fpr))
    auc_pr = float(trapezoid(precision, recall))
    return fpr, tpr, precision, recall, auc_roc, auc_pr


def plot_detector_usage(
    frame: Any,
    out_path: str | Path,
    detector_column: str = "detectors",
    title: str | None = None,
) -> Path:
    """绘制各检测器的调用次数（§17.4 策略层指标）。

    调用次数为零的检测器（或其判据永不触发）会非常醒目。

    Args:
        frame: 预测结果表（含检测器列，逗号分隔）。
        out_path: 输出图像路径。
        detector_column: 检测器列名。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / numpy。
        ValueError: 缺少列。
    """
    plt, zh = _prepare()
    import numpy as np

    if detector_column not in frame.columns:
        raise ValueError(f"缺少列 {detector_column!r}")

    counts: dict[str, int] = {}
    for value in frame[detector_column].fillna(""):
        for name in str(value).split(","):
            name = name.strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        raise ValueError("检测器列为空，无法统计调用次数")

    names = sorted(counts, key=lambda key: counts[key])
    values = [counts[name] for name in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names)), 4.2))
    bars = ax.bar(x, values, color="tab:brown", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom",
                fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(_t(zh, "调用次数", "Invocations"), fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        title or _t(zh, "检测器调用统计（§17.4）", "Detector usage (strategy-level metric)"),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_conflict_timeline(
    frame: Any,
    out_path: str | Path,
    conflict_column: str = "conflict",
    window: int = 200,
    title: str | None = None,
) -> Path:
    """绘制冲突率的滑动曲线（§12.4）。

    冲突率长期偏高说明检测器之间意见分歧大——通常意味着**判据区分度不足**，
    应优先修特征或阈值，而不是继续加检测器。

    Args:
        frame: 预测结果表（含冲突列）。
        out_path: 输出图像路径。
        conflict_column: 冲突列名（布尔）。
        window: 滑动窗口长度。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / pandas。
        ValueError: 缺少列或 window 非正。
    """
    plt, zh = _prepare()
    pd = _require_pandas()

    if conflict_column not in frame.columns:
        raise ValueError(f"缺少列 {conflict_column!r}")
    if window < 1:
        raise ValueError(f"window 必须 >= 1：{window}")

    series = frame[conflict_column].astype(bool).astype("float64")
    rolling = series.rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(range(len(rolling)), rolling.to_numpy(), color="tab:purple", linewidth=1.0)
    ax.axhline(float(series.mean()), linestyle="--", color="gray", linewidth=1,
               label=f"{_t(zh, '整体冲突率', 'overall')} = {float(series.mean()):.3f}")
    ax.set_xlabel(_t(zh, "历元序号", "Epoch index"), fontsize=10)
    ax.set_ylabel(_t(zh, "冲突率", "Conflict rate"), fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(
        title
        or _t(zh, f"冲突率滑动曲线（窗口 {window}）", f"Rolling conflict rate (window {window})"),
        fontsize=12,
    )
    return _save(fig, out_path)


def plot_correlation_matrix(
    frame: Any,
    columns: Sequence[str],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """绘制特征相关性热图（§6.3）。

    高相关的一对特征信息冗余，可用于精简特征集或解释“多检测器同时触发”。

    Args:
        frame: 特征表。
        columns: 参与计算的特征列。
        out_path: 输出图像路径。
        title: 图标题。

    Returns:
        写出的文件路径。

    Raises:
        ImportError: 未安装 matplotlib / pandas / numpy。
        ValueError: 可用列少于 2 个。
    """
    plt, zh = _prepare()
    pd = _require_pandas()
    import numpy as np

    available = [name for name in columns if name in frame.columns]
    if len(available) < 2:
        raise ValueError(f"至少需要 2 个可用特征列，实际 {len(available)}")

    corr = frame[available].corr(numeric_only=True).to_numpy(dtype="float64")
    size = max(6.0, 0.42 * len(available))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=90, fontsize=7)
    ax.set_yticklabels(available, fontsize=7)
    ax.set_title(title or _t(zh, "特征相关性热图", "Feature correlation matrix"), fontsize=12)
    return _save(fig, out_path)


def make_evaluation_figures(
    frame: Any,
    figures_dir: str | Path = "results/figures",
    prefix: str = "eval",
    feature_columns: Sequence[str] = (),
    truth_events: Sequence[Any] = (),
    predicted_events: Sequence[Any] = (),
    true_column: str = "Label",
    pred_column: str = "prediction",
    score_column: str = "confidence",
    detector_column: str = "detectors",
    conflict_column: str = "conflict",
) -> list[Path]:
    """一次性生成全部评估图件（供 ``scripts/evaluate.py`` / ``run_pipeline_check.py`` 调用）。

    生成内容（缺列或缺前置条件的项**自动跳过**，不影响其它图；跳过原因在返回值中不体现，
    但不会抛异常中断整批出图）：

    1.  ``*_timeline.png``        预测 / 真实 / 错误时段三联图；
    2.  ``*_confusion.png``       混淆矩阵（计数 + 行占比）；
    3.  ``*_class_metrics.png``   各类别 P/R/F1 柱状图；
    4.  ``*_label_dist.png``      真实 vs 预测的标签分布对比；
    5.  ``*_features.png``        关键特征曲线（需 ``feature_columns``）；
    6.  ``*_feature_box.png``     各特征按类别的箱线图（需 ``feature_columns``）；
    7.  ``*_feature_rank.png``    单特征区分度排序（需 ``feature_columns``）；
    8.  ``*_correlation.png``     特征相关性热图（需 ≥2 个特征列）；
    9.  ``*_confidence.png``      按真实标签分组的置信度分布（需 ``score_column``）；
    10. ``*_roc_pr.png``          ROC / PR 曲线（需 ``score_column`` 且真实标签含两类）；
    11. ``*_detector_usage.png``  检测器调用统计（需 ``detector_column``）；
    12. ``*_conflict.png``        冲突率滑动曲线（需 ``conflict_column``）；
    13. ``*_events.png``          事件时间线（需传入事件区间）。

    Args:
        frame: 预测结果表（须含 ``true_column`` 与 ``pred_column``）。
        figures_dir: 图件输出目录。
        prefix: 文件名前缀。
        feature_columns: 需要绘制曲线的特征列（可为空）。
        truth_events: 真实事件区间。
        predicted_events: 检测事件区间。
        true_column: 真实标签列。
        pred_column: 预测标签列。
        score_column: 置信度列（用于置信度分布与 ROC/PR）。
        detector_column: 检测器列（逗号分隔的检测器 id）。
        conflict_column: 冲突标记列。

    Returns:
        已写出文件的路径列表（可直接写入 §20.3 的 ``figure_paths``）。

    Raises:
        ImportError: 未安装 matplotlib / pandas。
        ValueError: 缺少必需列（``true_column`` / ``pred_column``）。
    """
    from src.eval.metrics import classification_report, confusion_matrix

    for column in (true_column, pred_column):
        if column not in frame.columns:
            raise ValueError(f"缺少列 {column!r}（现有列：{list(frame.columns)[:8]}…）")

    labels = tuple(sorted(CLASS_DISPLAY))
    directory = Path(figures_dir)
    outputs: list[Path] = []
    skipped: list[str] = []

    def _try(name: str, builder: Any) -> None:
        """尝试生成一张图；失败则记录原因并继续（不中断整批）。"""
        try:
            outputs.append(builder())
        except Exception as exc:  # noqa: BLE001 - 单张图失败不应影响其余
            skipped.append(f"{name}({type(exc).__name__})")

    available_features = [name for name in feature_columns if name in frame.columns]

    _try("timeline", lambda: plot_prediction_timeline(
        frame, directory / f"{prefix}_timeline.png",
        true_column=true_column, pred_column=pred_column))
    _try("confusion", lambda: plot_confusion_matrix(
        confusion_matrix(frame[true_column].tolist(), frame[pred_column].tolist(), labels=labels),
        directory / f"{prefix}_confusion.png"))
    _try("class_metrics", lambda: plot_class_metrics(
        classification_report(frame[true_column].tolist(), frame[pred_column].tolist(), labels=labels),
        directory / f"{prefix}_class_metrics.png"))
    _try("label_dist", lambda: plot_label_distribution(
        frame, directory / f"{prefix}_label_dist.png",
        true_column=true_column, pred_column=pred_column))

    if available_features:
        _try("features", lambda: plot_feature_timeline(
            frame, available_features, directory / f"{prefix}_features.png",
            label_column=true_column))
        _try("feature_box", lambda: plot_feature_distributions(
            frame, available_features, directory / f"{prefix}_feature_box.png",
            label_column=true_column))
        _try("feature_rank", lambda: plot_feature_discriminability(
            frame, available_features, directory / f"{prefix}_feature_rank.png",
            label_column=true_column))
        if len(available_features) >= 2:
            _try("correlation", lambda: plot_correlation_matrix(
                frame, available_features, directory / f"{prefix}_correlation.png"))

    if score_column in frame.columns:
        _try("confidence", lambda: plot_confidence_distribution(
            frame, directory / f"{prefix}_confidence.png",
            true_column=true_column, score_column=score_column))
        _try("roc_pr", lambda: plot_roc_pr_curves(
            frame, directory / f"{prefix}_roc_pr.png",
            score_column=score_column, true_column=true_column))

    if detector_column in frame.columns:
        _try("detector_usage", lambda: plot_detector_usage(
            frame, directory / f"{prefix}_detector_usage.png",
            detector_column=detector_column))

    if conflict_column in frame.columns:
        _try("conflict", lambda: plot_conflict_timeline(
            frame, directory / f"{prefix}_conflict.png",
            conflict_column=conflict_column))

    if truth_events or predicted_events:
        _try("events", lambda: plot_event_timeline(
            truth_events, predicted_events, directory / f"{prefix}_events.png"))

    if skipped:
        print(f"[plots] 跳过 {len(skipped)} 张图（条件不足）：{', '.join(skipped)}")
    return outputs
