"""M3 GNSS 上下文状态建模包。

对应开发文档
    §7 章（模块定位、Context 组成、上下文特征、上下文输出）、§18.1 Context 消融。

职责
    把多源特征压缩为结构化 Context（C_t = [S_t, Q_t, O_t, N_t, H_t, D_t]），
    作为数据层与策略决策层之间的唯一接口（§7.1）。

包含模块
    context_encoder：Context 数据类、分量访问、置信度与质量校验、编码器与消融开关

不做（边界）
    - 不做策略选择或检测器调用（→ src.agent）；
    - 不做检测与融合（→ src.detectors / src.fusion）。
"""
