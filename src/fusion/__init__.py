"""M6 多策略结果融合包。

对应开发文档
    §12 章（结果标准化、基础融合、融合规则、冲突处理）、§18.4 Multi-Strategy Fusion 消融。

职责
    把各检测器输出统一编码为 R_k = (y_k, p_k, e_k, t_k, q_k, c_k)，按置信度加权形成
    结构化判断，并显式暴露冲突供智能体决定是否追加验证。

包含模块
    result_fusion：StandardizedResult、FusedResult、fuse()、detect_conflict()、FUSION_ORDER

不做（边界）
    - 不再调用检测器（追加调用由 src.agent.executor 负责）；
    - 不做事件级状态管理（→ src.event）；
    - 不修改单检测器结果（§2.4）。
"""
