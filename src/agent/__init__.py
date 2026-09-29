"""M5 智能体动态策略选择包。

对应开发文档
    §9 章（决策输入、决策输出、动态决策流程、四原则、实现层级）、§8.4 Detector Registry。

职责
    按 Context、数据质量、历史结果与资源预算选择检测器组合，并完成工具调用编排；
    智能体只负责“调用谁、何时调用、是否追加验证”（§2.1）。

包含模块
    tool_registry：DetectorRegistry 与工具视图（§8.4、§9.6）
    planner      ：Plan 数据结构与 §8.3 矩阵路由
    policy       ：Level 0 固定策略 / Level 1 上下文规则路由，Level 2–3 预留
    executor     ：按计划执行、实测时延、隔离单点故障、支持提前终止

不做（边界）
    - 不实现检测算法（→ src.detectors）；
    - 不做融合与事件判定（→ src.fusion / src.event）；
    - 不修改检测结果（§2.4）。
"""
