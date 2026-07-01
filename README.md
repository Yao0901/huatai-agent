# huatai-agent

智谋洞见 — 基于 LangGraph ReAct + DeepSeek V4 的智能问数 Agent。

用户输入自然语言 → Agent 自主探索数据库结构 → 生成 SQL → 执行验证 → 自我修正 → 输出答案。

## 架构

```
用户输入
    ↓
ReAct Agent（单一 LLM 循环）
    ├── query_database      探索表结构 / 列名 / 枚举值 / 日期范围
    ├── search_code_mapping 查 dim_public 码值映射
    ├── run_sql             执行 SQL 并查看结果
    └── ask_user            不确定时追问用户澄清
    ↓
最终答案（含 SQL + 数据来源 + 解读）
```

不再使用固定流水线（3 张子图 × 8 个节点），全部由 LLM 自主决策。

## 项目结构

```
huatai-agent/
├── README.md
├── model/
│   ├── main.py                  # 入口（交互模式，支持多行输入）
│   ├── huatai.db                # SQLite 数据库（启动时从 CSV 自动构建）
│   ├── agents/
│   │   └── react_agent.py       # ReAct Agent：4 个工具 + System Prompt + 上下文管理
│   └── tools/
│       ├── db_connector.py      # SQLite 连接管理 + SQL 执行
│       ├── llm_config.py        # DeepSeek API（OpenAI 兼容 SDK + Token 计数 + ChatOpenAI 适配）
│       └── metric_retriever.py  # 口径定义匹配 + dim_public 码值查询
├── data/
│   └── 业务词汇匹配文件夹/
│       └── metric_definitions.json  # 8 条业务口径定义（名称、别名、描述、涉及表）
└── 01-金融大模型与智能体赛道-.../   # 原始 CSV 数据
```

## 快速开始

```bash
cd model

# 安装依赖
pip install langgraph langchain-openai sqlglot openai

# 配置 DeepSeek API Key（编辑 .env 文件，默认 deepseek-chat）
echo "DEEPSEEK_API_KEY=sk-xxx" > .env

# 交互模式
python main.py
```

## 示例

```
>>> 26年Q1日均资产大于30万的客户，股票交易量大于10万的，其持有的产品属于哪些产品大类

  ╔══════════════╗
  ║  容我三思   ║
  ╚══════════════╝
  💭 让我逐步分析，先探索相关表结构...
  💭 数据确认完毕，开始计算...
  📋 已了解: dwd_cust_tran_d, dwd_cust_hold_d, dim_product, dws_cust_aset_d

  ──────────────────────────────────────────────────
  📝 最终 SQL: WITH cust_avg_aset AS (...)
  ──────────────────────────────────────────────────

数据来源
   dws_cust_aset_d  客户日均资产
   dwd_cust_tran_d  股票交易流水
   dwd_cust_hold_d  客户持仓明细
   dim_product      产品大类映射

筛选过程
   日均资产 > 30万: 212 人
   股票交易 > 10万: 307 人
   交集: 179 人

结果: 7 个产品大类（股票、开放式基金、债券、私募、权证、理财、恒生多金融产品）
```

## 特性

- **自主探索**: LLM 自己查 PRAGMA、DISTINCT、样本，不需要人肉维护表结构文档
- **自我修正**: SQL 执行失败时自动分析原因并重试
- **Human-in-the-loop**: 口径/条件不确定时主动追问用户
- **上下文管理**: token 接近上限时智能修剪（保留首尾 + 重要消息，压缩闲聊）
- **思考可视化**: 「容我三思」标题 + 推理步骤 + 最终 SQL + 数据来源清单
- **纯文本输出**: 适配终端环境，不使用 Markdown 格式

## 当前已知缺口

1. **追问深度不足**: Agent 偶尔会自己猜而不是调用 `ask_user` 追问
2. **DeepSeek 编码相关**: `search_code_mapping` 查 dim_public 结果依赖 LIKE 模糊匹配
3. **缺少评测集**: 目标 >90% 准确率，尚无标准测试集
