"""
LLM 配置模块

负责从 .env 文件加载 DeepSeek API 配置，提供统一的 LLM 调用接口。
使用 OpenAI 兼容 SDK（openai 包），base_url 指向 DeepSeek。
"""

import json
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 环境变量加载（从 model/.env 文件）
# ---------------------------------------------------------------------------

def _load_env():
    """从 model/.env 文件加载环境变量（如果还没设置的话）。"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ or os.environ[key] == "your_api_key_here":
                        os.environ[key] = value


_load_env()


def get_llm_config() -> dict:
    """
    获取 LLM 配置字典。

    从环境变量读取 DeepSeek 配置，可用于初始化 ChatOpenAI 客户端。

    Returns:
        dict: {"api_key": str, "base_url": str, "model": str}

    Raises:
        ValueError: API Key 未设置时抛出
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "DeepSeek API Key 未设置。请在 model/.env 文件中填入你的 API Key。\n"
            "获取地址: https://platform.deepseek.com/api_keys"
        )

    return {
        "api_key": api_key,
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    }


# ---------------------------------------------------------------------------
# LLM 调用封装
# ---------------------------------------------------------------------------

# 全局客户端实例（懒初始化）
_client: Optional[object] = None


def _get_client():
    """获取 OpenAI 兼容客户端（懒初始化 + 单例）。"""
    global _client
    if _client is None:
        from openai import OpenAI

        config = get_llm_config()
        _client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
    return _client


def chat(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    向 DeepSeek 发送一次对话请求，返回模型的文本回复。

    这是整个项目中 LLM 调用的统一入口。所有节点（ambiguity_node、
    sql_gen_node、reflect_node 等）都通过此函数调用 LLM。

    Args:
        system_prompt: 系统提示词（角色设定 + 输出格式要求）
        user_message: 用户消息（具体的查询/分析内容）
        temperature: 生成温度，0.0=确定性强，1.0=高随机
        max_tokens: 最大输出 token 数

    Returns:
        str: 模型回复的文本

    Raises:
        ValueError: API Key 未配置
        Exception: API 调用失败时抛出
    """
    client = _get_client()
    config = get_llm_config()

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def chat_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_tool_rounds: int = 5,
) -> str:
    """
    向 DeepSeek 发送支持 tool calling 的对话请求。

    LLM 可以多次调用 tool，每次调用结果会追加到对话中，
    直到 LLM 不再请求 tool 或达到最大轮次。

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        tools: OpenAI 格式的 tool 定义列表
        temperature: 生成温度
        max_tokens: 最大输出 token 数
        max_tool_rounds: 最多 tool 调用轮次

    Returns:
        str: 最终的文本回复
    """
    client = _get_client()
    config = get_llm_config()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        msg = response.choices[0].message

        # 如果没有 tool_calls，返回文本
        if not msg.tool_calls:
            return msg.content or ""

        # 先追加 assistant 消息（含 tool_calls）
        messages.append(msg)

        # 执行每个 tool call，把结果追加为 tool 消息
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except Exception:
                func_args = {}

            # 解析并执行 tool
            result = _execute_tool(func_name, func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 超轮次，让 LLM 基于已有信息强制回复
    messages.append({
        "role": "user",
        "content": "请基于以上探索结果直接输出最终回答，不要再调用工具。",
    })
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Token 计数器
# ---------------------------------------------------------------------------

_token_stats = {"prompt": 0, "completion": 0, "total": 0}


def get_token_usage() -> dict:
    """返回当前会话的累计 token 消耗。"""
    return dict(_token_stats)


def reset_token_usage():
    """重置 token 计数。"""
    global _token_stats
    _token_stats = {"prompt": 0, "completion": 0, "total": 0}


from langchain_core.callbacks import BaseCallbackHandler


class _TokenCounter(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        try:
            usage = response.llm_output.get("token_usage", {})
            _token_stats["prompt"] += usage.get("prompt_tokens", 0)
            _token_stats["completion"] += usage.get("completion_tokens", 0)
            _token_stats["total"] = _token_stats["prompt"] + _token_stats["completion"]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LangChain ChatOpenAI 适配（供 create_react_agent 使用）
# ---------------------------------------------------------------------------

def get_chat_model(temperature: float = 0.0):
    """
    返回适配 DeepSeek 的 LangChain ChatOpenAI 实例。

    供 langgraph.prebuilt.create_react_agent 的 model 参数使用。
    """
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=temperature,
        callbacks=[_TokenCounter()],
    )


# ---------------------------------------------------------------------------
# Tool 执行器
# ---------------------------------------------------------------------------

def _execute_tool(func_name: str, args: dict) -> str:
    """
    执行 LLM 请求的 tool 调用。

    当前支持的 tools:
    - query_database: 执行只读 SQL 查询并返回结果
    """
    if func_name == "query_database":
        sql = args.get("sql", "")
        if not sql.strip():
            return "[ERROR] SQL 不能为空"

        # 安全检查：只允许 SELECT / PRAGMA
        sql_upper = sql.strip().upper()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA")):
            return f"[BLOCKED] 只允许 SELECT 和 PRAGMA 查询，不允许: {sql[:50]}"

        # 禁止查 sqlite_master（防止 LLM 瞎查系统表）
        if "sqlite_master" in sql_upper:
            return "[BLOCKED] 禁止查询 sqlite_master 系统表"

        try:
            from tools.db_connector import execute_sql
            result = execute_sql(sql)
            if result["success"]:
                rows = result["data"]
                if not rows:
                    return "[EMPTY] 查询无结果"
                # 限制返回行数
                if len(rows) > 100:
                    rows = rows[:100]
                    extra = f"\n... (还有 {result['row_count'] - 100} 行，已截断)"
                else:
                    extra = ""
                # 格式化输出
                cols = list(rows[0].keys())
                lines = [" | ".join(cols)]
                for row in rows:
                    lines.append(" | ".join(str(row.get(c, "")) for c in cols))
                return "\n".join(lines) + extra
            else:
                return f"[ERROR] {result.get('error', '未知错误')}"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    return f"[ERROR] 未知工具: {func_name}"
