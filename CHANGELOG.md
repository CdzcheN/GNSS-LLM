# CHANGELOG

> 本文件按《GNSS 多特征自适应干扰监测系统 项目开发文档 v1.1》**§28 文档维护规则**建立，
> 记录字段沿用其建议：`Version / Date / Author / Changed Modules / Reason / Experiment Impact`。
>
> **版本约定**
> - **规范版本**（开发文档正文）保持 **v1.1 不变**：本次工作是把文档已定义的接口与流程
>   首次落地为代码，未改变任何条款（§8.1 统一接口、§9.3 决策输出、§12.2 融合公式、
>   §13.3 状态机、§14.2 输入契约等均按原文实现）。
> - **实现版本** 从 `impl-0.1.0` 起独立编号，仅表示代码进展。

---

## [impl-0.4.0] 2026-09-29

**Author**：项目组（由 AI 编码助手生成，待负责人确认署名）

**Reason**：用户反馈“测试结果看不出问题”。诊断后发现两个根因并修复，同时新增阈值标定工具。

### 诊断结论（基于全量 42,932 行）

| 根因 | 证据 | 影响 |
|---|---|---|
| **数据划分错误（致命）** | 三类的时段不交错（Spoofing 12:32–16:44、Jamming 16:56–17:20、Normal 全天），按时间 70/30 切分后 **验证集只剩 Normal（12,880 行，0 个异常）** | 早停会挑出“全判正常”的模型——任何训练都无效 |
| **判据选错特征** | 区分度：原始列 `MaxRes` AUC **0.859**，而当时用的派生列 `res_valid_max` 仅 0.746；`cn0_delta_db` AUC 0.967 但阈值为经验值 −3.0 | Spoofing 召回仅有 0.271 |
| 阈值未标定 | 配置里多个键标注为“可运行起点” | 与文档 §16.3 的标定要求不符 |

### Changed Modules

| 模块 | 文件 | 变更 |
|---|---|---|
| 划分 | `src/train/dataset.py` | **新增** `find_event_segments` / `split_by_events` / `EventSplit` / `windows_from_segments`：按**连续同标签段**划分，每类的末尾若干段归验证集，**保证验证集含所有出现过的类别**；按段分别建窗，避免跨段拼接产生“假窗口” |
| 训练 | `src/detectors/deep_temporal.py` | `train()` 改为**优先事件级切分**（不可用时退化为时间切分并打印提示）；标准化统计量仍只取训练段；打印 train/val 的标签分布便于核对 |
| 检测器 | `src/detectors/observation.py` | S5 主判据改用**原始列 `MaxRes`**（新增 `FEATURE_MAX_RES`），`res_valid_max` 降为辅助 |
| 工具 | `scripts/calibrate_thresholds.py` | **新增**：在验证集上逐特征扫阈值（方向 × 数据分位点），输出最佳 F1 工作点、AUC、结论与可粘贴的 `config.yaml` 片段；NaN 按行剔除 |
| 配置 | `config.yaml` | 写入标定值：`cno.cn0_drop_db: −5.295`、`observation.residual_threshold: 176.7`、`pvt.clock_jump: 304`；标注 `satellite` 标定无效（AUC 0.667 / F1 0.038） |
| 测试 | `tests/test_calibration.py` | **新增** 18 个用例：事件段识别、划分不重叠且每类都有样本、按段建窗不跨段、AUC/阈值扫描的数值正确性、NaN 剔除 |

### 标定结果（验证集：事件级切分，含三类）

| 检测器 | 目标 | 最佳特征 | AUC | 阈值 | F1 |
|---|---|---|---|---|---|
| cno | Jamming | `cn0_delta_db` | 0.967 | ≤ −5.295 | 0.577 |
| observation | Spoofing | `MaxRes` | 0.859 | ≥ 176.7 | 0.571 |
| pvt | Spoofing | `clkB_diff` | 0.831 | ≥ 304 | 0.394 |
| satellite | Jamming | `valid_sat_count` | 0.667 | ≤ 5 | 0.038（几乎无效） |

### Experiment Impact

全量 42,932 行自检（非正式实验）：

| 指标 | 调优前 | 调优后 |
|---|---|---|
| Accuracy | 0.5622 | **0.7476** |
| Macro-F1 | 0.3332 | **0.4554** |
| Normal F1 | 0.691 | **0.841** |
| Spoofing Recall | 0.271 | **0.503** |
| Jamming F1 | 0（完全检不出） | 0.069 |
| 冲突率 | 0.1698 | **0.1469** |

**仍未解决的问题**：Jamming 召回偏低（全量 R=0.045，而验证集上单特征 R=0.913）——说明**事件间异质性显著**，
必须用 Leave-One-Event-Out（§16.2 的 19 个欺骗事件 / 10 个干扰事件）做交叉验证，固定单次划分会高估性能。

测试总数 222 → 240。

---

## [impl-0.3.1] 2026-09-29

**Author**：项目组（由 AI 编码助手生成，待负责人确认署名）

**Reason**：只有指标数字时看不出“检测在哪些时段出错、为什么出错”（尤其在预测全为 Normal 时）。
补充结果可视化，作为定位问题的第一手材料。

### Changed Modules

| 模块 | 文件 | 变更 |
|---|---|---|
| 可视化 | `src/eval/plots.py` | **新增**：预测/真实/错误时段三联时序图、混淆矩阵热图（计数+行占比）、各类别 P/R/F1 柱状图、关键特征曲线（叠加真实异常区间底色）、事件时间线、消融对比柱状图；Agg 无界面后端，中文字体尽力而为（无 CJK 字体时自动降级英文），所有函数返回写出的文件路径 |
| 入口 | `scripts/evaluate.py` | 新增 `--figures-dir` / `--feature-columns`，出图并把 `figure_paths` 写入 §20.3 实验记录，同时打印看图指引 |
| 入口 | `scripts/run_pipeline_check.py` | 自检增加第 6 步：自动生成图件并打印路径与看图要点 |
| 测试 | `tests/test_plots.py` | **新增** 13 个用例：五类图件与消融图均产出非空 PNG、`make_evaluation_figures` 打包产出、缺列/空输入报错、长序列抽样、无中文字体时不崩溃 |

### 用法

```bash
python -m scripts.run_pipeline_check --nrows 5000      # 自检并出图到 results/check/figures/
python -m scripts.evaluate --predictions results/pred.csv \
    --figures-dir results/figures --feature-columns cn0_delta_db,res_valid_mean
```

### 看图要点（用于定位“看不出问题”）

1. `*_timeline.png`：上=真实、中=预测、下=预测错误时段 → 错误是否集中在特定时段；
2. `*_confusion.png`：谁被误判成谁，行占比对小样本的 Jamming 尤其关键；
3. `*_class_metrics.png`：各类 P/R/F1，可直接看出 Jamming 是否被完全漏检；
4. `*_features.png`：关键特征曲线 + 真实异常区间底色。**若异常区间内特征无明显变化，
   说明该特征对当前攻击无区分度，问题不在检测阈值上，而应回到特征层（§6.3）**。

### Experiment Impact

无正式实验。可视化仅呈现既有结果，不改变任何指标或判定逻辑。
测试总数 197 → 210。

---

## [impl-0.3.0] 2026-09-29

**Author**：项目组（由 AI 编码助手生成，待负责人确认署名）

**Reason**：数据集已放入 `data/`，据此固化 §5.3 的列定义并补全 M1–M4 与 M9 的剩余实现，
把项目从「接口骨架」推进到「可用真实数据端到端运行」。

### 数据事实（实测，已固化为 schema 权威来源）

| 文件 | 覆盖 | 数据行数 |
|---|---|---|
| `gnss_complete_featuresObsSatPvt_12-16.csv` | 09-12 ~ 09-16 | 430,501 |
| `gnss_complete_featuresObsSatPvt_17-20.csv` | 09-17 ~ 09-20 | 343,828 |
| `gnss_complete_featuresObsSatPvt_21-25.csv` | 09-21 ~ 09-25 | 430,550 |
| `gnss_complete_featuresObsSatPvt_26-30.csv` | 09-26 ~ 09-30 | 430,477 |
| `gnss_complete_featuresObsSatPvtSpoofJamming.csv` | 12-21 | 42,932 |

- 列结构：112 列 = Timestamp/Day/Hour + 12 个 PVT 与全局特征 + 32 PRN × {CNO, Res, Elev} + Label，
  与开发文档 §5.3 模态 A 一致；有效数值特征 108 个。
- 三态标签分布实测 **0:33,602 / 1:8,714 / 2:616**，与 §5.2 记载的（33,847 / 8,604 / 481）不同
  ——当前数据集已更新，代码以实测为准（`src/data/schema.py:EXPECTED_LABEL_COUNTS`）。
- 正常数据扩展为 9 月 12–30 日共 19 天，故 §16.2 的划分按同一「时间连续、互不重叠」原则重划为
  **12–20 / 21–25 / 26–30**（与文件边界对齐），权威定义在 `src/data/schema.py:SPLIT_BY_DAY`。

### Changed Modules

| 模块 | 文件 | 变更 |
|---|---|---|
| M1 | `src/data/schema.py` | **新增**：列名与顺序、PRN 列表、时间格式、数据集清单、实测行数与标签分布、日期划分与自校验函数 |
| M1 | `src/data/parser.py` | 实现 CSV 读取 + schema 校验 + 时间戳解析 + 分块迭代 + 按划分过滤/标注 |
| M1 | `src/data/alignment.py` | 实现稳定排序、重复历元清理、多流 `merge_asof` 对齐（方向固定 backward，避免未来信息）、缺口报告与缺失时间点列表 |
| M1 | `src/data/quality.py` | 实现有效卫星数统计、`sat_mask`/`miss_mask`/`gap_before_s` 附加、可解释质量评分与分项明细 |
| M2 | `src/features/signal.py` | 实现 C/N0 统计与变化率、卫星数变化率（基线只用历史行） |
| M2 | `src/features/navigation.py` | 实现 PVT 精度与速度变化、DOP 滑动统计、`clkB`/`clkD` 一阶差分 |
| M2 | `src/features/observation.py` | 实现伪距残差统计、超限计数与 PRN 明细；多普勒/载波相位因数据未提供保留说明 |
| M2 | `src/features/spectrum.py` | 明确标注：当前数据集无 MON-SPAN 频谱/AGC 列，模态 B 特征不可实现 |
| M3 | `src/context/context_encoder.py` | 实现按列语义分派六分量、历史分量、置信度估计；**新增 `FlagThresholds` 与 `derive_flags`** 导出 §8.3 的环境状态判据；新增 `is_ready`（离线可用判据，D 为空属预期） |
| M4 | `src/detectors/base.py` | 新增 `numeric()` 安全取数工具 |
| M4 | `src/detectors/{cno,satellite,observation,pvt}.py` | 实现 S3/S4/S5/S6 判据（阈值全部取自 config，默认值标注为“未标定”） |
| M9 | `src/train/dataset.py` | 实现 `build_windows`（B×W×D，标签取窗口末行保证因果）；划分改为引用 schema |
| M9 | `src/train/trainer.py` | 实现 `fit`/`validate`（torch 惰性导入、梯度裁剪、早停、最优权重回滚、checkpoint 保存） |
| M9 | `src/detectors/deep_temporal.py` | 实现 `build_model`（LayerNorm + LSTM + 分类头 + 辅助头）、流式推理缓冲、`train`（训练集标准化随 checkpoint 保存，§6.4/§16.3） |
| M9 | `src/deploy/quantize.py` | 实现 `quantize_dynamic`（torch 动态量化） |
| 入口 | `scripts/{extract_features,build_context,train_detector,evaluate,run_agent}.py` | 全部接线为真实流程 |
| 入口 | `scripts/run_pipeline_check.py` | **新增**：一键端到端自检（数据→特征→Context→检测→融合→事件→指标） |
| 测试 | `tests/test_data_pipeline.py` | **新增**：25 个用例，用按 schema 生成的合成数据覆盖数据层/特征层/Context/检测器/窗口（不依赖真实数据集） |
| 配置 | `config.yaml` | `data` 段改为实际文件清单与采样参数；`splits` 改为日期区间；`detectors` 段补充各检测器阈值（**未标定**）；`event` 增加 `confirm_after_s` |

### 本轮发现并修复的缺陷

1. **特征层列组错位**（`src/features/observation.py`）：残差列组与 C/N0 列组名称不同，
   `DataFrame &` / `where` 会按列名取**并集**（32 列变 64 列）导致掩码全部错位、
   超限计数恒为 0。改为按位置（ndarray）对齐。
2. **策略路由永不生效**：`ContextRulePolicy` 依赖 `context.flags`，而 `Context` 无该字段，
   导致动态选择始终退化为兜底检测器（平均调用数恒为 1.00）。新增 `FlagThresholds` 与
   `derive_flags`，阈值与检测器判据同源。
3. `numeric()` 的 `isinstance(Mapping)` 检查过严，对 pandas Series 失效；改为 duck typing。
4. `slots=True` 的 dataclass 类属性是 slot 描述符，不能用作字段默认值读取；
   改为通过实例读取（`FlagThresholds.from_config`）。
5. `data_quality_score` 的缺口列回填顺序、`status` 字段缺失等若干实现细节。

### Experiment Impact

**无正式实验**。已完成的只是链路自检（`scripts/run_pipeline_check.py`）：
三态数据前 5,000 行跑通全链路，平均检测器调用 1.48、冲突率 0.17。
**注意**：各检测器阈值尚未标定（`config.yaml` 中标为可运行起点），自检所得指标
（Accuracy 0.56 / Macro-F1 0.33）仅用于判断链路是否通畅，**不能作为论文结果**。
正式指标须按 §16–§18 完成标定与消融后产生。

---

## [impl-0.2.0] 2026-09-27

**Author**：项目组（由 AI 编码助手生成，待负责人确认署名）

**Reason**：`impl-0.1.0` 把 §20.2 目录树中的 `src/train/`、`src/eval/`、`src/deploy/` 当作空目录占位，
未承载任何实现（用户指出 `src/train/` 只有 `__init__.py`）。本条目补齐这三个子包的实质代码，
并把被重复定义的指标键与实验记录字段收敛为**单一来源**。

### Changed Modules

| 模块 | 文件 | 变更 |
|---|---|---|
| 训练 | `src/train/reproducibility.py` | 新增：全局种子设定（numpy/torch 惰性）、多 seed 确定性派生、环境信息（§20.4） |
| 训练 | `src/train/losses.py` | 新增：类别权重推导（balanced 口径）与 §10.5 基线对账、LossSpec（§10.5） |
| 训练 | `src/train/dataset.py` | 新增：WindowConfig（限定 §6.2 敏感性集合）、SplitSpec、防泄漏校验、LOEO 规模（§6.2、§16.2、§16.3） |
| 训练 | `src/train/trainer.py` | 新增：TrainingConfig、EarlyStopping 状态机（已实现）、训练循环接口（§10.2、§22） |
| 训练 | `src/train/experiment.py` | 新增：config_hash（键序无关 SHA-256）、ExperimentRecord、CSV 追加写（§20.3） |
| 训练 | `src/train/unsupervised.py` | 新增：AutoEncoderConfig、两阶段预训练流程（§11.1、§11.2） |
| 评估 | `src/eval/metrics.py` | 新增：§17 五层指标键的**唯一来源**；混淆矩阵 / 逐类 P-R-F1 / Accuracy / Macro-F1 / 事件级匹配（已实现） |
| 评估 | `src/eval/ablation.py` | 新增：§18.1–§18.8 八个消融项及各自代码开关、对照组校验 |
| 部署 | `src/deploy/profiling.py` | 新增：时延分位数测量（预热 + 最近秩 p95）、参数量与 §1.4 G8 预算判定（已实现） |
| 部署 | `src/deploy/quantize.py` | 新增：§19.2 五步量化流程、相对精度损失与 < 1% 目标判定（未达标自动标注须如实报告） |
| 入口 | `scripts/evaluate.py` | 指标键、消融取值、实验记录字段改为引用单一来源（`src/eval/*`、`src/train/experiment.py`） |
| 包标记 | `src/train/__init__.py`、`src/eval/__init__.py`、`src/deploy/__init__.py` | 头部注释由“目录占位”改为模块清单与职责边界 |
| 测试 | `tests/{test_train,test_eval,test_deploy}.py` | 新增 68 个用例；测试总数 103 → 171 |

### 实现边界（新增部分仍未落地的内容）

均依赖 torch 或尚未固化的数据，已在各文件「待实现」段说明：
`weighted_cross_entropy`/`focal_loss`（需 torch）、`build_windows`（需 §5.3 列定义）、
`Trainer.fit/validate`、`build_autoencoder`/`reconstruction_error`、`quantize_dynamic`。

### Experiment Impact

**无**。仍未运行任何实验，`results/experiments_log.csv` 保持空模板。
新增的 `src/eval/metrics.py` 已具备计算逐秒/类别/事件级指标的能力，是后续首次实验结果记录的前置条件。

---

## [impl-0.1.0] 2026-09-27

**Author**：项目组（由 AI 编码助手生成，待负责人确认署名）

**Reason**：按开发文档 §20.2 建立工程骨架后，把 32 个占位文件补全为「详细头部注释 + 定型接口 +
可确定性实现即实现」的模块，并建立单元测试作为回归基线；同时补齐工程必需文件。

### Changed Modules

| 模块 | 文件 | 变更 |
|---|---|---|
| 工程根 | `config.yaml` | 新增：全局参数（窗口 60 s、LSTM 2×64、类别权重 1.0/3.9/70、focal γ=2、告警合并 30 s、seed 42 等），取值均来自文档既有条款 |
| 工程根 | `requirements.txt` | 新增：跨 Linux/Windows 的依赖清单（大版本区间约束，pip resolver 实测无冲突） |
| 工程根 | `.gitignore` | 新增：忽略缓存、数据与实验产物、模型权重、事件日志、会话目录 |
| 工程根 | `labels_1221.csv`、`results/experiments_log.csv` | 新增：表头模板（后者字段取自 §20.3） |
| 契约层 | `src/detectors/base.py` | **新增文件**：`AttackType`(§1.3)、`DetectionResult`(§8.1)、`DetectorMeta`(§8.4)、`BaseDetector` |
| M1 数据 | `src/data/{parser,alignment,quality}.py` | 定型签名与字段常量；`sat_mask=(CNO>0.5)` 按 §5.4 Q3 实现 |
| M2 特征 | `src/features/{signal,spectrum,observation,navigation}.py` | 按模态 A/B/C 定型；含 §6.3 的 clkB/clkD 差分、128 维谱表示 |
| M3 上下文 | `src/context/context_encoder.py` | `Context` 六分量 S/Q/O/N/H/D 与 §18.1 消融开关**已实现** |
| M4 检测器 | `src/detectors/{threshold,spectrum,cno,satellite,observation,pvt,deep_temporal}.py` | S1 配置驱动规则引擎**已实现**；S2–S7 接口与注册元信息定型 |
| M5 智能体 | `src/agent/{tool_registry,planner,policy,executor}.py` | 注册表、§8.3 矩阵路由、Level 0/1 策略、执行器**已实现** |
| M6 融合 | `src/fusion/result_fusion.py` | §12.2 置信度加权与 §12.4 冲突判定**已实现** |
| M7 事件 | `src/event/{event_record,event_manager,event_logger}.py` | §13.3 状态机、§13.2 合并窗口、JSONL/CSV 日志与告警**已实现** |
| M8 LLM | `src/llm/{prompt,schema,report,summarizer}.py` | §14.2 输入校验、§14.3 报告模板、降级摘要**已实现** |
| 入口 | `scripts/*.py`（5 个） | argparse 与流程注释齐备；`run_agent --dry-run` 可跑通全链路装配 |
| 测试 | `tests/*.py`（9 个） | **新增目录**：99 个单元测试（含头部注释规范与配置-代码一致性校验） |
| 文档 | `docs/experiment-protocol.md` | 新增：§16–§18、§20.3 的实验约束检查表 |

### 实现边界（尚未落地的部分）

共 87 处 `NotImplementedError`，均**必须依赖尚不可得的信息**，已在各文件头部「待实现」段逐条说明：

- 各检测器的阈值/判据标定（需标注数据与 §5.3 列定义固化）；
- M2 特征列名与 §5.3 的 112 列对应关系；
- LSTM 模型构建、训练与推理（需数据与 checkpoint）；
- 真实 LLM 提供方接入（§14.5 可插拔接口）。

### 与 §20.2 推荐目录的差异（有意保留）

1. **新增** `src/detectors/base.py` —— §8.1 统一接口需要一处载体，文档目录树未列出；
2. **新增** `tests/` —— 单元测试目录，文档目录树未列出；
3. **未创建** `docs/project-development-v1.1.md` —— 仓库根目录已有同内容文档，避免产生两份会漂移的副本（经项目负责人确认保持现状）；
4. 另有工程必需补充：各包 `__init__.py`、空目录 `.gitkeep`、`.gitignore`。

### Experiment Impact

**无**。本次变更不涉及数据划分、标签、判据阈值或模型超参的确定，未运行任何实验；
`results/experiments_log.csv` 仍为空模板（仅表头）。后续一旦固化 §5.3 列定义或标定判据阈值，
须按 §28 重新评估是否触发**规范版本**升级（涉及数据集、实验划分或核心模型变化时）。
