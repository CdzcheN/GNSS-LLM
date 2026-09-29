"""M9 训练子包：可复现性、损失、数据划分、训练循环与实验记录。

对应开发文档
    §10 深度时序检测模型、§11 无监督与半监督辅助路线、§16 实验设计与防泄漏、
    §20.3 实验记录规范、§20.4 固定随机种子、§22 风险管理（早停与正则）。

职责
    承载训练流程的全部实现；``scripts/train_detector.py`` 只是命令行入口，业务逻辑位于本包。

包含模块
    reproducibility：全局随机种子设定与多 seed 确定性派生（§20.4）
    losses         ：类别加权 CE / Focal Loss 配置与类别权重推导（§10.5）
    dataset        ：划分协议、窗口配置与防泄漏校验（§6.2、§16.2、§16.3）
    trainer        ：训练配置、早停状态机与训练循环接口（§10.2、§22）
    experiment     ：ExperimentRecord、config_hash 与 CSV 追加写（§20.3）
    unsupervised   ：Mask-aware LSTM AutoEncoder 与两阶段流程（§11.1、§11.2）

不做（边界）
    - 训练不得使用测试集参与 scaler 拟合或特征选择（§16.3）；
    - 不得对连续时间序列做无约束随机打散（§2.5）；
    - 模型结构定义在 ``src/detectors/deep_temporal.py``，本包只负责训练与记录。
"""
