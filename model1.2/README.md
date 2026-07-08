# huatai-agent 1.2

华泰证券智能问数原型系统 —— 单 ReAct Agent + Schema 缓存 + 多行粘贴输入。

## 一句话概括

> 用户用自然语言提问 → LLM 自主探索数据库结构 → 生成 SQL → 执行 → 看结果 → 修正 → 输出答案。

## 数据库

SQLite（Python 标准库 `sqlite3`），启动时从 `data/` 目录递归扫描 CSV 自动建表导入。8 张表：

| 表名 | 内容 |
|---|---|
| ads_cust_info_d | 客户信息（等级、性别、年龄、学历、职业） |
| dws_cust_aset_d | 客户资产日汇总（普通账户 / 富裕账户） |
| dwd_cust_tran_d | 客户交易流水（买入/卖出金额与笔数） |
| dwd_cust_hold_d | 客户持仓明细（持仓数、市值） |
| dws_cust_fin_d | 客户资金流入流出 |
| dim_product | 产品维度 |
| dim_branch | 营业部维度 |
| dim_public | 码值映射 |

**SQL 方言：SQLite**（Python 标准库 `sqlite3` 3.45.3）。后续规划支持 DuckDB 等多方言执行引擎。

## Agent 权限

| 层级 | 限制 | 实现位置 |
|---|---|---|
| `run_sql` | 只允许 SELECT / WITH，拦截 INSERT/UPDATE/DELETE/DROP | `tools/run_sql.py` |
| `query_database` | 只允许 SELECT / PRAGMA，拦截 sqlite_master | `tools/query_database.py` |
| `execute_sql` | 单连接 `check_same_thread=False`，异常统一捕获返回错误不抛出 | `tools/db_connector.py` |
| 结果行数 | `run_sql` 最多返回 30 行，`query_database` 最多 50 行，超出截断 | 各自工具文件 |
| LLM API | 通过 `.env` 配置，不硬编码密钥 | `tools/llm_config.py` |

Agent 无写数据库能力，无文件系统访问能力，不能访问网络。

## 功能特性

- **ReAct 自愈循环**：SQL 执行失败 → LLM 分析错误 → 查表确认 → 修正 → 重试
- **Schema 缓存**：表结构、枚举值、日期范围、行数各只查一次，跨轮复用
- **上下文修剪**：对话过长时保留前 3 轮 + LLM 生成中段摘要 + 最近 10 轮
- **多轮对话记忆**：MemorySaver 持久化会话历史，追问不需要重新探索
- **流式思考展示**：实时显示 LLM 推理过程（「容我三思」→ 思考 → 最终答案）
- **多行粘贴**：prompt_toolkit 终端，Ctrl+Enter 发送，Enter 换行，Bracketed Paste 自动识别
- **纯文本输出**：适配终端环境，用【】和缩进组织内容，不依赖 Markdown
- **输出规范**：最终回答包含 [SQL] / [数据来源] / [关键数据] / [解读] / [本轮摘要]

## 项目结构

```
model1.2/
├── main.py              # 入口：启动初始化 + 流式交互循环
├── agent/
│   └── react_agent.py   # ReAct Agent 构建（System Prompt + 工具注册 + 上下文修剪）
└── tools/
    ├── db_connector.py      # SQLite 连接 + CSV 自动导入 + execute_sql()
    ├── llm_config.py        # LLM API 配置（OpenAI 兼容）
    ├── query_database.py    # 数据库结构探索（带 Schema 缓存）
    ├── run_sql.py           # SELECT 执行 + 安全校验 + 结果格式化
    ├── ask_user.py          # 向用户追问澄清
    └── schema_cache.py      # 内存缓存（表结构/枚举值/日期范围/行数）
```