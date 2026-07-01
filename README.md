# huatai-agent

华泰证券 Agentic 智能问数 — 基于 LangGraph + DeepSeek 的 Text-to-SQL 原型系统。

用户输入自然语言取数问题 → 意图识别 → 表结构检索 → SQL 生成 → 安全评估 → 执行返回结果。

## 项目结构

```
huatai-agent/
├── README.md
├── model/
│   ├── main.py                  # 唯一入口（交互模式 / 单次查询）
│   ├── huatai.db                # SQLite 数据库（启动时从 CSV 自动构建）
│   ├── agents/
│   │   ├── state.py             # 全局状态 TypedDict 定义
│   │   ├── main_graph.py        # 主图：串联 intent → query → security 三张子图 + 跨图重试
│   │   ├── intent_agent.py      # 子图1：意图识别（歧义检测 → 口径检索 → 追问消歧）
│   │   ├── query_agent.py       # 子图2：查询生成（LLM选表 → 贴合度校验 → 追问 → 生成SQL）
│   │   └── security_agent.py    # 子图3：安全评估（sqlglot分级 → 危险确认 → 权限注入 → 执行 → 反思）
│   ├── nodes/
│   │   ├── ambiguity_node.py    # 节点3：LLM 判断用户输入是否存在模糊口径/目标
│   │   ├── metric_search_node.py# 节点3.1：LLM 抽取关键词 → 向量检索口径定义 + 码值映射
│   │   ├── schema_node.py       # 节点4：组装检索上下文 → 调用 schema_retriever 召回表结构
│   │   ├── sql_gen_node.py      # 节点6：DeepSeek 生成 SQL（含表结构+口径+码值+跨轮历史）
│   │   ├── security_node.py     # 节点7/7.1：sqlglot AST 安全分级 + 行级权限过滤注入
│   │   ├── execution_node.py    # 节点8：SQLite 执行 SQL，返回结构化结果或报错
│   │   └── reflect_node.py      # 节点9：LLM 诊断执行报错 → 生成修正建议 → 回退重试
│   └── tools/
│       ├── db_connector.py      # SQLite 连接管理：初始化建库、CSV 导入、schema 查询、SQL 执行
│       ├── llm_config.py        # DeepSeek API 统一入口（OpenAI 兼容 SDK）
│       ├── metric_retriever.py  # RAG 两段式口径匹配：sentence-transformers 向量粗筛 + LLM 精判
│       ├── schema_retriever.py  # 表结构检索：LLM 语义选表（优先）+ 关键词匹配（兜底）+ 关联扩散
│       └── ast_parser.py        # sqlglot AST 解析：安全分级、提取表/列、权限条件注入
├── data/
│   └── 业务词汇匹配文件夹/
│       ├── metric_definitions.json  # 8 条业务口径定义（别名、公式、涉及表）
│       ├── metric_vectors.json      # 口径向量库（每条指标一条 centroid 向量）
│       ├── rebuild_vectors.py       # 一键从 definitions.json 重建向量库
│       └── model/                   # 本地 sentence-transformers 中文模型
└── 01-金融大模型与智能体赛道-.../   # 原始 CSV 数据（9 张维度建模表）
```

## 当前已知缺口

### 1. 表结构元数据缺少离散列枚举值

`tools/schema_retriever.py` 的 `TABLE_DESCRIPTIONS` 已包含每列的**中文描述**和**JOIN 关系**，但未标注离散列的**合法取值**。例如：

| 列 | 枚举数 | 风险 |
|---|---|---|
| `prdt_type_name` | 60+ 种（A股、科创板、ETF…） | LLM 可能生成 `WHERE prdt_type_name='美股'`（不存在） |
| `prof_cd` | 59 种职业编码 | LLM 不知道有哪些可选值 |
| `cust_lvl_cd` | 6 种客户等级 | 值域已知但未注入列描述 |
| `gender_cd` | 2 种有效值 | 风险较低但建议标注 |

此外 `dim_public` 的 `code_type_id` 还有 200、300、400 三种类型未在 `CODE_TYPE_MAP` 中覆盖，用途待确认。

> 详见记忆：`column-enum-values-todo`

### 2. 缺乏推理流程可视化

当前执行过程是黑盒——用户输入问题，等几秒后看到结果（或报错）。无法观察：

- LLM 选表时的**候选表和得分**
- 口径检索的**向量相似度 top-k 和 Stage 2 精判过程**
- SQL 生成的**完整 prompt 和原始响应**
- 安全评估的 **AST 分级决策**
- 重试时**反思节点的诊断内容**

建议后续加入中间状态展示（如 `--verbose` 模式或 Web 调试面板）。

### 3. 缺乏日志记录

没有任何日志持久化。全部输出靠 `print()` 到终端，退出后不可追溯。建议补充：

- 用户问题 / 生成 SQL / 执行结果 / 耗时 / 错误信息 的结构化日志
- 至少按日期分文件，便于后续构建评测集和排查问题

### 4. 跨轮上下文管理不完整

`main.py` 的交互循环已维护 `conversation_history`（最近 5 轮），并注入到 LLM 节点的 prompt 中，但仍有不足：

- **追问多轮状态**: Agent1 的 `ask_user_node` 通过追加 `（补充：...）` 到 `user_input` 来累积上下文，多轮追问后 `user_input` 会越来越长，无截断/摘要机制
- **记忆读写**: 项目 memory 文件在 `C:\Users\Administrator\.claude\projects\D--HuaTai-agent\memory\`，但 Agent 本身不读写这些记忆——记忆仅由 Claude Code 在开发过程中使用，Agent 运行时无记忆持久化

### 5. SQL 执行缺少保护

`execution_node.py` 直接执行 SQL，没有超时限制和行数上限。`dim_product` 表 33 万行、`dwd_cust_hold_d` 表 40 万行，`SELECT *` 会撑爆内存。

### 6. 缺少评测体系

目标 >90% 准确率，但没有测试集。需要 50-100 条 Q&A pair 覆盖简单查询到复杂多表 JOIN、歧义消解、时间过滤等场景，用于回归验证。

## 快速开始

```bash
cd model
pip install sqlglot sentence-transformers openai

# 配置 DeepSeek API Key（编辑 .env 文件）
echo "DEEPSEEK_API_KEY=sk-xxx" > .env

# 交互模式
python main.py

# 单次查询
python main.py --query "30岁以下女性客户的持仓市值"
```
