"""M1 多源数据接入与解析包。

对应开发文档
    §5 数据资产与数据管理、§6.1 标准处理流水线（前四步）。

职责
    把 Raw/Processed 原始数据转换为时间对齐、带质量掩码的可计算记录表。

包含模块
    parser    ：JSON / UBX 解析与时间戳标准化
    alignment ：时间排序与多流对齐（§5.4 Q1、Q2）
    quality   ：sat_mask / miss_mask 与数据质量评分（§5.4 Q2、Q3）

不做（边界）
    - 不做特征工程（→ src.features）；
    - 不做标准化（§6.4 要求统计量仅来自训练集）；
    - 不做任何检测判定（→ src.detectors）。
"""
