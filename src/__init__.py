"""GNSS 多特征自适应干扰监测系统 —— 源码根包。

对应开发文档
    §4.1 系统分层（M1–M9）、§20.2 推荐目录、§27 最终设计原则总结。

职责
    汇聚九大模块的实现：
        src.data       M1 多源数据接入与解析
        src.features   M2 特征联合表征
        src.context    M3 GNSS 上下文状态建模
        src.detectors  M4 检测策略库（S1–S7）
        src.agent      M5 智能体动态策略选择
        src.fusion     M6 多策略结果融合
        src.event      M7 异常事件管理与实时上报
        src.llm        M8 LLM 日志总结与报告生成
        src.train / src.eval / src.deploy  M9 训练、评估与部署性能监测

不做（边界）
    - 本包只做模块聚合与契约约束，不承载业务逻辑；
    - 各模块职责严格按 §2.1 检测与决策解耦，不互相越界。
"""
