# huatai-agent

智谋洞见 — 基于 LangGraph ReAct的智能问数 Agent。

用户输入自然语言 → Agent 自主探索数据库结构 → 生成 SQL → 执行验证 → 自我修正 → 输出答案。

## 使用说明

### 1. 准备数据

将赛方提供的 CSV 数据文件夹复制到 `data/` 下，启动时 `main.py` 会自动读取 CSV 建库。

### 2. 进入 model 目录

```bash
cd model1.2
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 LLM API

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key 并取消注释对应厂商的配置行。

### 5. 启动

```bash
python main.py
```

交互式对话，输入 SQL 问数需求，输入 `exit` 退出。


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

