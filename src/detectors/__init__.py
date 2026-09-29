"""M4 检测策略库包（S1–S7）。

对应开发文档
    §8 章（统一接口、策略分类、适用性矩阵、Detector Registry）、§10 深度时序检测模型。

职责
    提供统一接口的检测器实现，全部继承 ``base.BaseDetector``，供注册表编排调用。

包含模块
    base          ：统一接口与结构化结果契约（§8.1）
    threshold     ：S1 固定阈值/统计检测（配置驱动，兼作兜底路径）
    spectrum      ：S2 频谱检测
    cno           ：S3 C/N0 检测
    satellite     ：S4 卫星状态检测
    observation   ：S5 观测一致性检测
    pvt           ：S6 PVT/DOP 检测
    deep_temporal ：S7 深度时序检测（多源融合 + LSTM + 三分类头）

不做（边界）
    - 不做策略选择与执行编排（→ src.agent）；
    - 不做结果融合与冲突处理（→ src.fusion）；
    - 检测器之间不得互相改写对方输出（§2.4）。
"""
