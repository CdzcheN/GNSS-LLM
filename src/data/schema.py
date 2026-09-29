"""数据 schema：把实际数据集的 112 列表头固化为可校验契约。

对应开发文档
    §5.1 原始数据、§5.2 已提取数据、§5.3 模态定义（模态 A 的 112 列结构）、
    §6.1 标准处理流水线、§16.2 数据划分。

职责
    1. 固化列名与分组：时间（Timestamp/Day/Hour）+ PVT 与全局状态（12 列）
       + 32 个 PRN × {CNO, Res, Elev}（96 列）+ Label；
    2. 固化数据集文件清单、日期覆盖范围与实测行数（用于回归验收）；
    3. 固化本仓库实际采用的训练/验证/留出划分（按日期，时间连续且互不重叠）；
    4. 提供列名与划分的自校验函数，避免上游数据改动被静默忽略。

不做（边界）
    - 不读取数据文件（只做定义与校验，I/O 属 ``src/data/parser.py``）；
    - 不做特征构造（→ ``src/features/*``）与标准化（§6.4 统计量只能来自训练集）；
    - 不实现随机划分（§2.5、§16.3 明令禁止）。

输入 / 输出
    输入：列名序列、日期值
    输出：校验结果、列名查询、划分查询

关键约束
    - 列名**与顺序**必须与 ``data/`` 下实际表头一致（本文件由实测表头固化）；
    - 数值特征列数必须为 108（112 − Timestamp/Day/Hour/Label）；
    - 划分必须按日期连续切分且互不重叠；训练集不得包含验证或留出日。
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- 基础列

#: 时间戳列（实测格式 ``2023-09-12 00:00:00``）。
TIME_COLUMN: str = "Timestamp"
#: 日期列（正常数据为 12–30；三态数据为字符串 ``1221``）。
DAY_COLUMN: str = "Day"
#: 小时列（0–23）。
HOUR_COLUMN: str = "Hour"
#: 标签列（§1.3 三态）。
LABEL_COLUMN: str = "Label"

#: 时间戳解析格式（与实测表头内容一致）。
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"

#: PVT 与全局状态列（12 列，§5.3 模态 A）。
PVT_COLUMNS: tuple[str, ...] = (
    "hAcc",    # 水平精度
    "vAcc",    # 垂直精度
    "tAcc",    # 时间精度
    "clkB",    # 钟偏（§6.3 要求一阶差分）
    "clkD",    # 钟漂（§6.3 要求一阶差分）
    "gSpeed",  # 地速
    "pDOP",    # 位置精度因子
    "tDOP",    # 时间精度因子
    "hDOP",    # 水平精度因子
    "NumSats",  # 可见卫星数
    "AvgCNO",  # 平均载噪比
    "MaxRes",  # 最大伪距残差
)

#: 卫星 PRN 列表（GPS，G01–G32；§5.3 记载为 “32 PRN”）。
PRN_LIST: tuple[str, ...] = tuple(f"G{index:02d}" for index in range(1, 33))

#: 每个 PRN 对应的三类观测列后缀。
PER_PRN_SUFFIXES: tuple[str, ...] = ("CNO", "Res", "Elev")


def prn_column(prn: str, suffix: str) -> str:
    """按 PRN 与后缀拼接列名。

    Args:
        prn: 卫星编号，如 ``"G01"``。
        suffix: 观测类型，取值见 ``PER_PRN_SUFFIXES``。

    Returns:
        列名，如 ``"CNO_G01"``。

    Raises:
        ValueError: 后缀非法。
    """
    if suffix not in PER_PRN_SUFFIXES:
        raise ValueError(f"未知观测后缀 {suffix!r}，允许：{PER_PRN_SUFFIXES}")
    return f"{suffix}_{prn}"


#: 观测列（96 列）。顺序与实测表头一致：按 PRN 交错（CNO_G01, Res_G01, Elev_G01, CNO_G02, …）。
OBSERVATION_COLUMNS: tuple[str, ...] = tuple(
    prn_column(prn, suffix) for prn in PRN_LIST for suffix in PER_PRN_SUFFIXES
)

#: 载噪比列（32 个）。
CNO_COLUMNS: tuple[str, ...] = tuple(prn_column(prn, "CNO") for prn in PRN_LIST)
#: 伪距残差列（32 个）。
RES_COLUMNS: tuple[str, ...] = tuple(prn_column(prn, "Res") for prn in PRN_LIST)
#: 卫星高度角列（32 个）。
ELEV_COLUMNS: tuple[str, ...] = tuple(prn_column(prn, "Elev") for prn in PRN_LIST)

#: 完整列顺序（112 列），与实测表头逐列一致。
EXPECTED_COLUMNS: tuple[str, ...] = (
    TIME_COLUMN,
    DAY_COLUMN,
    HOUR_COLUMN,
    *PVT_COLUMNS,
    *OBSERVATION_COLUMNS,
    LABEL_COLUMN,
)

#: 可用于计算的数值特征列（108 列 = 112 − 时间 3 列 − 标签 1 列）。
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (*PVT_COLUMNS, *OBSERVATION_COLUMNS)

#: 期望列数（§5.3 的 112 列与 108 维有效数值特征）。
EXPECTED_COLUMN_COUNT: int = len(EXPECTED_COLUMNS)
EXPECTED_FEATURE_COUNT: int = len(NUMERIC_FEATURE_COLUMNS)

# --------------------------------------------------------------------------- 标签

#: 三态标签取值（§1.3；与 ``src.detectors.base.AttackType`` 一致，由测试断言保证）。
LABEL_NORMAL: int = 0
LABEL_SPOOFING: int = 1
LABEL_JAMMING: int = 2

#: 标签 → 名称。
LABEL_NAMES: Mapping[int, str] = {
    LABEL_NORMAL: "Normal",
    LABEL_SPOOFING: "Spoofing",
    LABEL_JAMMING: "Jamming",
}

# --------------------------------------------------------------------------- 数据集清单

#: 数据集目录（相对仓库根）。
DATASET_DIR: str = "data"

#: 正常数据文件（9 月 12–30 日，19 天）。
NORMAL_FILES: tuple[str, ...] = (
    "gnss_complete_featuresObsSatPvt_12-16.csv",
    "gnss_complete_featuresObsSatPvt_17-20.csv",
    "gnss_complete_featuresObsSatPvt_21-25.csv",
    "gnss_complete_featuresObsSatPvt_26-30.csv",
)

#: 三态数据文件（12-21 欺骗/干扰实验）。
THREE_STATE_FILE: str = "gnss_complete_featuresObsSatPvtSpoofJamming.csv"

#: 正常数据的日期覆盖范围（含端点）。
NORMAL_DAY_RANGE: tuple[int, int] = (12, 30)

#: 三态数据 ``Day`` 列的取值（字符串标记，非日期数字）。
THREE_STATE_DAY: str = "1221"

#: 各文件的数据行数（不含表头），实测值，用于回归验收。
EXPECTED_ROWS: Mapping[str, int] = {
    "gnss_complete_featuresObsSatPvt_12-16.csv": 430_501,
    "gnss_complete_featuresObsSatPvt_17-20.csv": 343_828,
    "gnss_complete_featuresObsSatPvt_21-25.csv": 430_550,
    "gnss_complete_featuresObsSatPvt_26-30.csv": 430_477,
    "gnss_complete_featuresObsSatPvtSpoofJamming.csv": 42_932,
}

#: 三态数据的标签分布（实测值）。注意：与开发文档 §5.2 记载的
#: （33,847 / 8,604 / 481）不同——当前数据集已更新，代码以实测为准。
EXPECTED_LABEL_COUNTS: Mapping[int, int] = {
    LABEL_NORMAL: 33_602,
    LABEL_SPOOFING: 8_714,
    LABEL_JAMMING: 616,
}

# --------------------------------------------------------------------------- 划分

#: 按日期的数据划分（§16.2 的时间段划分精神：时间连续、互不重叠、禁止随机打散）。
#:
#: 开发文档 §16.2 给出的是「12–13 → Train、14 → Validation、15 → Hold-out」，
#: 对应当时 4 天的数据；当前数据集扩展为 9 月 12–30 日共 19 天，因此按同一精神
#: 重划，并保持与文件边界对齐（便于按文件加载）。
SPLIT_BY_DAY: Mapping[str, tuple[int, ...]] = {
    "train": tuple(range(12, 21)),       # 12–20，对应 _12-16 与 _17-20 两个文件
    "validation": tuple(range(21, 26)),  # 21–25，对应 _21-25
    "holdout": tuple(range(26, 31)),     # 26–30，对应 _26-30
}

#: 划分名称。
SPLIT_NAMES: tuple[str, ...] = tuple(SPLIT_BY_DAY)


def assert_columns(columns: Sequence[str]) -> None:
    """校验列名与 schema 完全一致（含顺序）。

    Args:
        columns: 实际列名序列。

    Raises:
        ValueError: 存在缺失列、多余列，或顺序不一致。
    """
    actual = list(columns)
    expected = list(EXPECTED_COLUMNS)
    if actual == expected:
        return

    actual_set, expected_set = set(actual), set(expected)
    missing = [name for name in expected if name not in actual_set]
    extra = [name for name in actual if name not in expected_set]
    if missing or extra:
        raise ValueError(
            f"列名与 schema 不一致：缺少 {missing[:5]}（共 {len(missing)}），"
            f"多出 {extra[:5]}（共 {len(extra)}）"
        )
    raise ValueError("列名集合一致但顺序不同：顺序是索引约定的一部分，须与表头一致")


def assert_feature_count(columns: Iterable[str]) -> None:
    """校验数值特征列数为 108（§5.3）。

    Args:
        columns: 待校验的数值特征列。

    Raises:
        ValueError: 数量与期望不符。
    """
    count = len(list(columns))
    if count != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"数值特征列数应为 {EXPECTED_FEATURE_COUNT}（§5.3），实际为 {count}"
        )


def days_for_split(name: str) -> tuple[int, ...]:
    """返回某划分包含的日期。

    Args:
        name: ``"train"`` / ``"validation"`` / ``"holdout"``。

    Returns:
        日期元组。

    Raises:
        KeyError: 划分名不存在。
    """
    try:
        return SPLIT_BY_DAY[name]
    except KeyError as exc:
        raise KeyError(f"未知划分 {name!r}，允许：{SPLIT_NAMES}") from exc


def split_of_day(day: int | str) -> str:
    """判断某个日期属于哪个划分。

    Args:
        day: 日期值（``Day`` 列内容）。

    Returns:
        划分名称。

    Raises:
        ValueError: 该日期不属于任何划分（可能落入了未参与训练的时间段）。
    """
    if isinstance(day, str):
        digits = "".join(ch for ch in day if ch.isdigit())
        # 三态文件的 Day 为 "1221" 形式，单独标记
        if day == THREE_STATE_DAY:
            return "event_1221"
        if not digits:
            raise ValueError(f"无法解析日期值：{day!r}")
        day = int(digits)
    for name, days in SPLIT_BY_DAY.items():
        if int(day) in days:
            return name
    raise ValueError(f"日期 {day} 不属于任何划分（{SPLIT_BY_DAY}）")


def assert_split_disjoint() -> None:
    """自检：各划分的日期集合互不相交。

    Raises:
        ValueError: 存在跨划分重叠。
    """
    seen: dict[int, str] = {}
    for name, days in SPLIT_BY_DAY.items():
        for day in days:
            if day in seen:
                raise ValueError(f"日期 {day} 同时属于 {seen[day]!r} 与 {name!r}")
            seen[day] = name


def column_groups() -> Mapping[str, tuple[str, ...]]:
    """返回列名分组，供特征与 Context 编码复用。

    Returns:
        分组名 → 列名元组。
    """
    return {
        "time": (TIME_COLUMN, DAY_COLUMN, HOUR_COLUMN),
        "pvt": PVT_COLUMNS,
        "cno": CNO_COLUMNS,
        "res": RES_COLUMNS,
        "elev": ELEV_COLUMNS,
        "label": (LABEL_COLUMN,),
    }
