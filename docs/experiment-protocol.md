# 实验协议（Experiment Protocol）

> 本文件是《GNSS 多特征自适应干扰监测系统 项目开发文档 v1.1》第 16–18、20.3 章的落地检查表。
> 参数取值统一以仓库根目录 `config.yaml` 为准；本文件只记录**必须遵守的流程约束**。

## 1. 实验体系（开发文档 16.1）

| 编号 | 机制 | 目的 |
|---|---|---|
| Baseline-1 | 固定阈值 | 评价传统固定判据 |
| Baseline-2 | 固定规则多特征融合 | 评价静态多源融合 |
| Baseline-3 | 静态 ML/DL | 评价单模型能力 |
| Method-1 | 上下文感知策略选择 | 验证动态选择机制 |
| Method-2 | Agent 多策略协同 | 验证完整智能系统 |

## 2. 数据划分（16.2）

- **正常数据实验**：12–13 → Train，14 → Validation，15 → Hold-out。
- **1221 实验**：一期基线可用前 70% / 后 30% 时间顺序切分；正式动态策略与泛化实验**必须优先按攻击事件划分**。
- **Leave-One-Event-Out**：按 19 个欺骗事件区间、10 个干扰事件区间执行跨事件留出测试。

## 3. 防泄漏规范（16.3）

**禁止**：随机窗口打散 / 测试集参与 scaler fit / 测试集参与特征选择 / 测试事件信息反向影响策略选择器训练。

**允许**：预定义时间段 / 事件级切分 / 跨事件验证。

## 4. 评价指标（17 章）

| 层级 | 指标 |
|---|---|
| 逐秒级 | Accuracy、Precision、Recall、F1、Macro-F1、AUC-ROC、AUC-PR、FPR |
| 类别级 | Spoofing P/R/F1、Jamming P/R/F1（**Jamming 必须单独报告**） |
| 事件级 | Event Detection Rate、Event Recall、Detection Delay、False Alarm Events、Missed Events、Event-level Precision |
| 策略层 | 平均调用检测器数量、深度模型调用率、平均策略决策时间、平均检测器执行时间、策略冲突率、二次验证触发率 |
| 系统级 | 端到端检测延迟、数据处理吞吐、内存占用、模型参数量、INT8 精度变化、LLM 总结延迟、告警可用率 |

> 结果优先于指标：不以单一 Accuracy / 单一 F1 判断系统效果（2.6）。

## 5. 消融与对比清单（18 章）

- [ ] 18.1 Context 消融：无 Context vs 有 Context
- [ ] 18.2 Dynamic Selection 消融：固定策略 vs 动态策略选择
- [ ] 18.3 Agent 消融：Rule Router vs Learning Router vs Agent Router
- [ ] 18.4 Multi-Strategy Fusion 消融：单检测器 vs 多检测器不融合 vs 多检测器融合
- [ ] 18.5 Deep Detector 消融：无深度模型 vs 调用深度模型
- [ ] 18.6 模态消融：A vs A+B vs A+C vs A+B+C
- [ ] 18.7 窗口敏感性：10 / 30 / 60 / 120 s
- [ ] 18.8 LLM 模块验证：事实一致性、字段完整性、摘要可读性、生成延迟、结构化字段与文本是否一致

## 6. 记录规范（20.3 / 20.4）

每个实验必须在 `results/experiments_log.csv` 记录：
`experiment_id, config_hash, model_version, data_version, split_version, seed, metrics, figure_paths, checkpoint_path, conclusion`。

- 基线默认 `seed = 42`；研究性实验可多 seed 重复并报告均值与标准差。
- 禁止修改测试集相关配置后复用同一 `experiment_id`。

## 7. 待定项（尚未在本仓库确定）

- `labels_1221.csv` 的最终字段与事件区间来源（当前仅含占位表头 `timestamp,label`）。
- 事件区间清单（19 个欺骗 / 10 个干扰）的固化位置。
- `splits/` 下划分文件的命名与格式。
