"""M7 异常事件管理包。

对应开发文档
    §13 章（事件定义、告警合并、事件状态机）、§15.1 在线流程与实时告警、
    §14.4 C4 保留原始日志。

职责
    把逐秒（逐窗口）判定聚合为事件级记录，驱动状态机，并输出结构化日志与实时告警。

包含模块
    event_record ：EventRecord、EventState、状态迁移表、as_llm_input 桥接
    event_manager：告警合并窗口（§13.2）与状态机（§13.3）
    event_logger ：JSONL/CSV 落盘、回读与告警文本

不做（边界）
    - 事件状态不得由 LLM 生成（§13.3）；状态只能来自结构化检测逻辑；
    - 不做检测与融合（→ src.detectors / src.fusion）。
"""
