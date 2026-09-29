# 后续开发与测试顺序（Development & Test Roadmap）

> 依据：《项目开发文档 v1.1》§16–§21、`CHANGELOG.md` 的 `impl-0.2.0` 实际实现状态。
> 编排原则：**先补全代码 → 同步模块测试 → 最后参数优化**，且每一步都要给出可执行的验收判据。
> 版本：`plan-1`（2026-09-27）

---

## 0. 起点：现状与唯一的硬阻塞

### 0.1 已完成（可运行且有测试）

| 层 | 状态 |
|---|---|
| 契约层 `src/detectors/base.py` | ✅ `AttackType` / `DetectionResult` / `DetectorMeta` / `BaseDetector` |
| M4 策略库 | ✅ S1 阈值规则引擎可运行；S2–S7 接口与注册元信息定型 |
| M5 智能体 | ✅ 注册表、§8.3 矩阵路由、Level 0/1 策略、执行器（时延实测 + 故障隔离 + 提前终止） |
| M6 融合 | ✅ §12.2 置信度加权、§12.4 冲突判定 |
| M7 事件 | ✅ §13.3 状态机、§13.2 合并窗口、JSONL/CSV 日志、告警文本 |
| M8 LLM | ✅ §14.2 契约校验、§14.3 报告模板、降级摘要（未接真实模型） |
| M9 指标/消融 | ✅ §17 混淆矩阵与 P/R/F1/Macro-F1、事件级匹配；§18 八个消融定义 |
| M9 训练脚手架 | ✅ 种子管理、类别权重推导、防泄漏校验、早停、`config_hash` 与实验记录 |
| M9 部署测量 | ✅ 时延分位数、参数量预算（§1.4 G8）、INT8 损失判定（§19.2） |
| 回归基线 | ✅ `python3 -m unittest discover -s tests -t .` → 171 用例全绿 |

### 0.2 硬阻塞（其余工作的前置）

> **§5.3 的 112 列定义未固化。**

`data/` 目前为空，`src/data/parser.py` 无法把实际 CSV 列映射到语义组，
于是**特征 → Context → 检测器 → 训练 → 评估**整条链路都无法用真实数据验证。
因此 **A1 是所有后续工作的唯一入口**，其余任务之间才是依赖关系。

### 0.3 待补全规模（按开发文档章节聚合）

| 章节 | 占位数量 | 主要内容 |
|---|---|---|
| §6.3 派生特征 | 10 | 信号/频谱/观测/导航四类特征 |
| §6.1 / §5.x 数据层 | 5 + 2 + 3 + 1 | 解析、对齐、掩码、时间戳 |
| §5.3 模态 B/C | 3 + 1 | 频谱与 RAWX 特征 |
| §11.1 无监督 | 3 | 自编码器与异常分数 |
| §7.3 / §7.4 Context | 1 + 2 + 1 | 质量评分、Context 编码、批量编码 |
| §8.2 S2–S6 | 5 | 五个规则检测器判据 |
| §10.2 / §10.3 / §10.5 | 2 + 1 + 2 | LSTM 构建、训练、损失 |
| §9.6 Level 2/3、§14.5、§15.1、§17–§19 | 各 1–2 | 学习型策略、LLM 接入、流水线、量化 |

（完整清单可用 `grep -rn "NotImplementedError" src scripts --include=*.py` 复核。）

---

## 1. 总体顺序

```text
A 补全代码  ──依赖数据流从上游到下游──▶  每完成一项立即接 B 的对应测试
   A1 schema（唯一硬阻塞）
   A2 解析 → A3 对齐 → A4 质量掩码
   A5–A8 特征（信号/导航/观测/频谱）
   A9 Context 编码
   A10–A15 规则检测器（S3→S4→S5→S6→S2→S1 接线）
   A16–A17 深度模型与训练
   A18–A21 学习型策略 / LLM / 量化 / 脚本接线

B 模块测试 ──与 A 交替进行，不排到最后──▶  B1–B10

C 参数优化 ──必须等 B 全绿、数据链路跑通后──▶  C1–C10
```

**顺序理由**

1. 检测器的输入是特征，特征的输入是解析结果 → 上游未固化前写下游只能靠臆造；
2. 参数优化会改动判据阈值，若代码未定稿，标定结果立刻作废；
3. 测试与实现交替，可让每个模块在"刚写完"时就被固定行为，避免缺陷向后传播。

---

## 2. 阶段 A：补全代码（逐项）

> 每项的「完成判据」都是可观测事实，优先使用开发文档已给出的数字（数据集行数、缺失时长等）。

### A1 固化数据 schema

| 项 | 内容 |
|---|---|
| 任务 | 读取 `GNSSInterfer/gnss_complete_featuresObsSatPvt12-15.csv` 实际表头，把 112 列映射到 §5.3 模态 A 的组别（Timestamp / Day / Hour / PVT / 全局状态 / 32 PRN × CNO,Res,Elev / Label） |
| 产出 | `src/data/schema.py`（或 `parser.py` 顶部常量）：列名→语义组映射、`recordTime` 格式、有效数值列数（应为 108） |
| 前置 | **无**（唯一无前置项）；需数据文件就位（`data/` 当前为空） |
| 完成判据 | 列总数 = 112；`Label` 列存在于 1221 数据；`sat_mask`/`miss_mask` 的目标列名确定 |
| 验收 | 新增 `tests/test_schema.py`：断言列数、组别覆盖、关键列齐备 |
| 章节 | §5.2、§5.3、§6.1 |

### A2 实现数据解析

| 项 | 内容 |
|---|---|
| 任务 | 实现 `parse_processed_json` / `parse_ubx_stream` / `parse_many` / `standardize_timestamp` |
| 文件 | `src/data/parser.py` |
| 前置 | A1 |
| 完成判据 | 正常数据解析后**数据行数 = 344,361**（§5.2：344,362 含表头）；1221 数据 = **42,932** 行；`Label` 分布为 Normal 33,847 / Spoofing 8,604 / Jamming 481 |
| 验收 | `tests/test_parser.py` |
| 章节 | §5.1、§5.2、§6.1 |

### A3 实现时间排序与多流对齐

| 项 | 内容 |
|---|---|
| 任务 | 实现 `sort_by_time`、`align_streams`、`alignment_report` |
| 文件 | `src/data/alignment.py` |
| 前置 | A2 |
| 完成判据 | 排序后时间戳单调不减；对齐后**缺口总时长 = 1,239 s**（§5.2 给出的缺失量）；`AlignmentReport.max_gap_s` 非空 |
| 验收 | `tests/test_alignment.py`（含"对齐不得使用未来历元"的确定性检查） |
| 章节 | §5.4 Q1、Q2、§6.1 |

### A4 实现质量掩码与质量评分

| 项 | 内容 |
|---|---|
| 任务 | 实现 `miss_mask`、`data_quality_score`、`attach_masks`（`sat_mask` 已实现） |
| 文件 | `src/data/quality.py` |
| 前置 | A3 |
| 完成判据 | `sat_mask` 统计与 §5.4 Q3 一致；质量评分落在 [0,1]；正常数据质量 > 干扰区间质量 |
| 验收 | 扩充 `tests/test_context_and_quality.py` |
| 章节 | §5.4 Q2/Q3、§7.3 |

### A5 信号特征（模态 A/B）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `cn0_features`、`agc_features`、`signal_features` |
| 文件 | `src/features/signal.py` |
| 前置 | A4 |
| 完成判据 | C/N0 变化率窗口 = 60 s（§6.2）；无卫星历元被 `sat_mask` 屏蔽后再统计；压制区间的 C/N0 下降量可复算 |
| 验收 | `tests/test_features_signal.py`（含手算样例） |
| 章节 | §6.3、§7.3 |

### A6 导航特征（PVT/DOP/钟差）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `pvt_features`、`dop_features`、`clock_features`、`navigation_features` |
| 文件 | `src/features/navigation.py` |
| 前置 | A4 |
| 完成判据 | `clkB`/`clkD` 一阶差分列存在且命名与 §6.3 一致；DOP 滑动统计窗口 = 60 s |
| 验收 | `tests/test_features_navigation.py` |
| 章节 | §6.3、§7.3 |

### A7 观测一致性特征（模态 C）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `pseudorange_residual`、`doppler_consistency`、`carrier_consistency`、`observation_features` |
| 文件 | `src/features/observation.py` |
| 前置 | A4（RAWX 数据就位） |
| 完成判据 | 按 PRN 与历元可追溯；周跳被标注而非混入一致性统计；证据中保留 PRN 明细 |
| 验收 | `tests/test_features_observation.py` |
| 章节 | §5.3 模态 C、§6.3 |

### A8 频谱特征（模态 B）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `spectrum_statistics`、`spectrum_embedding`、`spectrum_features` |
| 文件 | `src/features/spectrum.py` |
| 前置 | A4（MON-SPAN 数据就位） |
| 完成判据 | 谱表示维度固定 128（§5.3）；统计量可与 AGC 联动核对（用于排除增益变化） |
| 验收 | `tests/test_features_spectrum.py` |
| 章节 | §5.3 模态 B、§6.3 |

### A9 Context 编码

| 项 | 内容 |
|---|---|
| 任务 | 实现 `ContextEncoder.encode` / `encode_batch`（含分量裁剪与置信度估计） |
| 文件 | `src/context/context_encoder.py` |
| 前置 | A5–A8 |
| 完成判据 | 正常区间 `is_complete = True`；`enabled=False` 时返回 `None`（§18.1 对照组）；质量分与 §7.4 的 Data Quality 一致 |
| 验收 | 扩充 `tests/test_context_and_quality.py` |
| 章节 | §7.2、§7.4、§18.1 |

### A10–A13 规则检测器（按"证据最确定"到"最不确定"排序）

| 项 | 检测器 | 前置 | 完成判据 | 章节 |
|---|---|---|---|---|
| A10 | `src/detectors/cno.py`（S3 C/N0） | A5 | 481 s Jamming 区间召回可复算；无卫星历元不产生结论 | §8.2 S3、§8.3 |
| A11 | `src/detectors/satellite.py`（S4 卫星状态） | A6 | 卫星增减在证据中列出 PRN 明细 | §8.2 S4 |
| A12 | `src/detectors/observation.py`（S5 观测一致性） | A7 | 超限残差列出 PRN 与残差值；周跳不被判为欺骗 | §8.2 S5 |
| A13 | `src/detectors/pvt.py`（S6 PVT/DOP） | A6 | 位置/速度突变与 DOP 变化给出具体数值与单位 | §8.2 S6 |

### A14 频谱检测器（S2）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `SpectrumDetector.run`（判据与 AGC 联动） |
| 文件 | `src/detectors/spectrum.py` |
| 前置 | A8 |
| 完成判据 | 与 S3 组合时能覆盖 §8.3 的"频谱异常明显"场景；阈值来源可追溯到标定结果 |
| 验收 | `tests/test_detector_spectrum.py` |
| 章节 | §8.2 S2、§8.3 |

### A15 S1 阈值检测器接线

| 项 | 内容 |
|---|---|
| 任务 | 把 C1 标定出的判据写入 `config.yaml`（**不硬编码**），并作为兜底路径接入 `run_agent` |
| 前置 | A10–A14、C1（可先给临时值，标定后回填） |
| 完成判据 | `--policy fixed` 下系统仍能独立产出告警（§9.5 原则 D） |
| 验收 | `tests/test_agent.py` 扩充 |
| 章节 | §8.2 S1、§9.5 |

### A16 深度时序模型

| 项 | 内容 |
|---|---|
| 任务 | 实现 `build_model`（§10.3 多源融合 + LSTM + 三分类头）与 `run`（加载 checkpoint 推理） |
| 文件 | `src/detectors/deep_temporal.py` |
| 前置 | A2–A9（张量构造） |
| 完成判据 | 参数量 < 10⁶（§1.4 G8，用 `src/deploy/profiling.count_parameters` 核验）；无 checkpoint 时返回 `SKIPPED` 而非崩溃 |
| 验收 | `tests/test_detector_deep.py`（结构 + 参数量，不依赖训练） |
| 章节 | §10.2、§10.3 |

### A17 训练链路

| 项 | 内容 |
|---|---|
| 任务 | 实现 `dataset.build_windows`、`losses.weighted_cross_entropy`/`focal_loss`、`Trainer.fit`/`validate`；`unsupervised` 的 `build_autoencoder`/`reconstruction_error`/`anomaly_score` |
| 前置 | A16 |
| 完成判据 | 可复现训练：同一 seed 两次运行得到相同 `val_loss` 序列；早停生效；实验记录自动追加到 `results/experiments_log.csv` |
| 验收 | `tests/test_train_pipeline.py`（小样本冒烟） |
| 章节 | §10.5、§11.1、§11.2、§22 |

### A18 学习型策略（可选，后置）

| 项 | 内容 |
|---|---|
| 任务 | 实现 `LearningPolicy`（Level 2）与 `AgentPolicy`（Level 3） |
| 前置 | A17、C5（需要足够实验数据） |
| 完成判据 | §18.3 的 Rule Router / Learning Router / Agent Router 三方对比可运行 |
| 章节 | §9.6、§19.3（有限工具集与决策步数） |

### A19 LLM 接入

| 项 | 内容 |
|---|---|
| 任务 | 实现 `LLMSummarizer.summarize`（异步、只读结构化输入） |
| 前置 | A21（需真实事件流） |
| 完成判据 | 摘要字段与结构化记录逐项一致；LLM 不可用时不阻塞告警（`TemplateSummarizer` 自动接管） |
| 章节 | §14.4、§15.2、§19.4 |

### A20 量化实现

| 项 | 内容 |
|---|---|
| 任务 | 实现 `quantize_dynamic`，接入 §19.2 五步流程 |
| 前置 | A16、A17 |
| 完成判据 | 量化前后指标与体积写入 `QuantizationReport`；未达标时报告自动标注"须如实报告" |
| 章节 | §19.2 |

### A21 脚本接线（端到端收口）

| 项 | 内容 |
|---|---|
| 任务 | 用真实实现替换 5 个脚本的 `NotImplementedError`：`extract_features` → `build_context` → `train_detector` → `evaluate` → `run_agent` |
| 前置 | A1–A20 |
| 完成判据 | `python -m scripts.run_agent` 能在 1221 数据上跑完整闭环，产出事件 JSONL 与实验记录行 |
| 验收 | `tests/test_pipeline_e2e.py`（小切片，秒级完成） |
| 章节 | §15.1、§20.2 |

---

## 3. 阶段 B：模块测试（与 A 交替）

> 原则：**每个模块完成即测**，不把测试攒到最后；所有测试必须能在无 GPU、无网络的 CI 上秒级运行。

| 项 | 测试对象 | 数据 | 关键断言 | 触发时机 |
|---|---|---|---|---|
| B1 | 数据层（parser/alignment/quality） | 12–15 与 1221 | 行数 344,361 / 42,932；缺口 1,239 s；`sat_mask=(CNO>0.5)` | A2–A4 完成后 |
| B2 | 特征层（signal/navigation/observation/spectrum） | 同上 | 手算样例逐值比对；列名与 §6.3 命名一致 | A5–A8 各自完成时 |
| B3 | Context | 同上 | 六分量齐备；质量分区间；`enabled=False` 返回 `None` | A9 后 |
| B4 | 各检测器行为 | 1221 的三态切片 | 逐检测器输出可与人工复核的证据（PRN/数值明细） | A10–A14 各自完成时 |
| B5 | 训练链路 | 小样本 | 同 seed 可复现；早停按耐心值触发；checkpoint 可加载 | A17 后 |
| B6 | 融合与冲突 | 构造结果 | §12.4 文档示例必须判为冲突 | 已在跑，随 A16 扩充 |
| B7 | 事件层端到端 | 合成逐秒流 | 事件切分、合并窗口 30 s、状态迁移全部可断言 | 已在跑，A21 后接真实流 |
| B8 | **防泄漏专项** | 划分定义 | 训练/验证/测试无交集；scaler 只在训练集拟合；事件级划分无跨集事件 | A3、A17 后**必须**做 |
| B9 | LLM 契约与一致性 | 固定事件 | 白名单校验；摘要文本与结构化字段逐项一致（§18.8） | A19 后 |
| B10 | 系统级冒烟 | 小切片 | `evaluate` 能产出五层指标并写入实验记录 | A21 后 |

**B8 是硬性门槛**：§16.3 明令禁止的四种泄漏（随机打散、测试集参与 scaler、测试集参与特征选择、测试事件影响策略训练）
必须各有至少一条自动化断言。

---

## 4. 阶段 C：参数优化（逐项，且必须最后做）

> 铁律：**所有标定只在验证集上进行**，hold-out 与 LOEO 结果只在最终评估时使用一次（§16.2、§16.3）。
> 每项调参都必须写入 `results/experiments_log.csv`（§20.3），含 `config_hash` 与 `seed`。

| 项 | 调什么 | 在哪调 | 范围/选项 | 判据 | 章节 |
|---|---|---|---|---|---|
| C1 | 各规则检测器判据阈值 | `config.yaml` 的 `detectors` 段（S1 走 `rules` 列表） | 以验证集 ROC/PR 曲线上的工作点为准 | 单检测器 F1 与误报率的折中；阈值必须可追溯 | §8.2、§17.2 |
| C2 | 检测窗口长度 | `data.window_s` | 10 / 30 / 60 / 120 s（§6.2） | 综合检测性能、时延、上下文判断与开销 | §6.2、§18.7 |
| C3 | 类别权重与 focal γ | `class_imbalance` | 权重由训练集计数推导（基准 1.0/3.9/70）；γ = 0/1/2/5 | Jamming 召回提升且 Normal 误报不显著恶化 | §10.5、§17.2 |
| C4 | 模型超参 | `TrainingConfig` | 层数 1–3、hidden 32/64/128、lr、batch | 验证集 Macro-F1 与参数量（< 10⁶）双约束 | §10.2、§1.4 G8 |
| C5 | 早停与正则 | `TrainingConfig` | patience、`min_delta`、`l2_weight_decay`、梯度裁剪 | 跨事件表现不掉点（§22 过拟合信号） | §22 |
| C6 | 融合权重 | `fusion` 段 | 数据质量、检测器可靠度、开销的组合 | 冲突率下降而召回不降 | §12.2、§12.4 |
| C7 | 策略选择门限与适用性 | `agent` 段、`src/agent/planner.py` 的矩阵 | 提前终止置信度门限、`max_detectors`、`max_decision_steps` | 平均检测器调用数下降而性能不降（§9.5 原则 A） | §8.3、§9.4、§9.5 |
| C8 | Context 分量取舍 | `src/context/context_encoder.py` 分量开关 | §18.1 的"无 Context vs 有 Context" | 动态选择相对固定策略有可重复收益 | §18.1、§18.2 |
| C9 | 量化参数 | `deploy` 段 | 量化方式（动态/静态）、校准集规模 | 精度损失 < 1%（不达标须如实报告） | §19.2 |
| C10 | 多 seed 重复 | `src/train/reproducibility.seed_sequence` | 至少 3 个 seed | 报告均值与标准差，而非单点最优 | §20.4 |

**C 阶段的常见错误（需显式避免）**

1. 用测试集挑阈值 → 判据作废；
2. 只报最优 seed → 违反 §20.4；
3. 调参后不更新 `config_hash` → 实验记录不可追溯；
4. 把 LLM 摘要质量混入算法指标（§18.8 要求单列）。

---

## 5. 里程碑与验收卡点

| 里程碑 | 包含 | 验收证据（可执行） |
|---|---|---|
| **M-A1 数据链路可跑** | A1–A4、B1 | `python -m unittest tests.test_parser tests.test_alignment` 通过；行数与缺口数等于文档值 |
| **M-A2 特征与 Context 就绪** | A5–A9、B2–B3 | 特征列与 §6.3 命名一致；`Context.is_complete` 在正常区间为 True |
| **M-A3 规则检测器可用** | A10–A15、B4 | `run_agent --policy fixed` 在 1221 上产出非空事件流 |
| **M-A4 深度模型可训可推** | A16–A17、B5 | 参数量 < 10⁶；同 seed 重跑 `val_loss` 一致；实验记录新增一行 |
| **M-A5 端到端闭环** | A18–A21、B6–B8、B10 | 一次完整回放产出事件 JSONL + 实验记录；**B8 防泄漏断言全绿** |
| **M-C1 参数标定完成** | C1–C8 | 每个实验在 `results/experiments_log.csv` 有一行且 `config_hash` 唯一 |
| **M-C2 部署评估完成** | A20、C9 | `QuantizationReport` 与参数量/时延数据齐备，结论明确 |

---

## 6. 并行机会与风险

### 可并行

- A5（信号）与 A6（导航）互不依赖，可同时进行；
- A7、A8 在 RAWX / MON-SPAN 数据就位后可并行；
- B1–B3 的测试骨架可在 A 进行时先行搭建（用合成数据）。

### 风险与对策

| 风险 | 触发信号 | 对策 |
|---|---|---|
| A1 拖延 | 数据文件缺失或列义不清 | **优先解决**：可先用 §5.2 的列数/行数约束做契约测试，再补语义映射 |
| 模态 B/C 数据缺失 | 无法实现 A7/A8 | 先做 A（仅模态 A）的完整闭环，模态消融（§18.6）后置 |
| 检测器判据标定不出可用阈值 | C1 后单检测器 F1 接近随机 | 退回检查特征质量（B2），而非继续调阈值 |
| 深度模型过拟合 | 跨事件掉点 | 启用 C5 的早停与正则；必要时先做 §11 的无监督预训练 |
| 动态选择无收益 | C7/C8 与固定策略差异不显著 | 按 §22 的预案加强策略适用性建模与分环境实验，**如实报告** |

---

## 7. 每项工作的完成标准（Definition of Done）

任一 A/B/C 条目完成时，须同时满足：

1. **代码**：该模块无遗留 `NotImplementedError`，或已在文件头部「待实现」段说明原因与依赖；
2. **测试**：新增对应用例，且 `python3 -m unittest discover -s tests -t .` 全绿；
3. **配置**：新增参数进入 `config.yaml`，且 `tests/test_config.py` 的一致性断言通过；
4. **记录**：`CHANGELOG.md` 追加条目（字段：Version / Date / Author / Changed Modules / Reason / Experiment Impact，见开发文档 §28）；
5. **可追溯**：涉及判据或超参的，必须在 `results/experiments_log.csv` 留一行。
