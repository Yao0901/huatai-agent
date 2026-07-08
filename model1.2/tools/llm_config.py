"""
LLM 配置模块

负责从 .env 文件加载 LLM API 配置，提供统一的 LLM 调用接口。
使用 OpenAI 兼容 SDK（openai 包），base_url 指向配置的 LLM 服务。
"""

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 环境变量加载（从 model1.2/.env 文件）
# ---------------------------------------------------------------------------

def _load_env():
    """从 model1.2/.env 文件加载环境变量（如果还没设置的话）。"""
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

    从环境变量读取 LLM 配置（LLM_API_KEY / LLM_BASE_URL / MODEL_ID），
    兼容旧版 DEEPSEEK_* 变量名，可用于初始化 ChatOpenAI 客户端。

    Returns:
        dict: {"api_key": str, "base_url": str, "model": str}

    Raises:
        ValueError: API Key 未设置时抛出
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "LLM_API_KEY 未设置。请在 model1.2/.env 文件中填入你的 API Key。"
        )

    return {
        "api_key": api_key,
        "base_url": os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", ""),
        "model": os.environ.get("MODEL_ID") or os.environ.get("DEEPSEEK_MODEL", ""),
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
    max_tokens: int = 8192,
) -> str:
    """
    向 LLM 发送一次对话请求，返回模型的文本回复。

    这是整个项目中 LLM 调用的统一入口。

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


# ---------------------------------------------------------------------------
# LangChain ChatOpenAI 适配（供 create_react_agent 使用）
# ---------------------------------------------------------------------------

def get_chat_model(temperature: float = 0.0):
    """
    返回适配 LLM 服务的 LangChain ChatOpenAI 实例。

    供 langgraph.prebuilt.create_react_agent 的 model 参数使用。
    """
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=temperature,
    )
