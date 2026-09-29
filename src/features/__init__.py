"""M2 多源特征联合表征包。

对应开发文档
    §5.3 模态定义（A 基础状态 / B AGC 与频谱 / C RAWX 观测）、
    §6.3 派生特征、§7.3 上下文特征。

职责
    按模态构造可解释特征：信号、频谱、观测一致性、导航解算。

包含模块
    signal      ：C/N0、AGC 及其变化率（模态 A/B）
    spectrum    ：频谱统计与 128 维谱表示（模态 B）
    observation ：伪距 / 多普勒 / 载波一致性（模态 C）
    navigation  ：PVT、DOP 滑动统计、clkB / clkD 一阶差分

不做（边界）
    - 不做阈值判别与结论输出（→ src.detectors）；
    - 不做标准化与滑窗张量构造（→ scripts/extract_features.py，§6.1、§6.2）。
"""
