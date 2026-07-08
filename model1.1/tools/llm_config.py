"""
LLM 配置模块（精简版）

负责从 .env 文件加载 LLM API 配置，提供 LangChain ChatOpenAI 实例。
仅保留 ReAct Agent 需要的 get_chat_model()。
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 环境变量加载
# ---------------------------------------------------------------------------

def _load_env():
    """从 .env 文件加载环境变量。优先 model1.1/.env，其次 model/.env。"""
    for env_name in [".env", "../model/.env"]:
        env_path = Path(__file__).parent.parent / env_name
        resolved = env_path.resolve()
        if resolved.exists():
            with open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip()
                        if key not in os.environ or os.environ[key] == "your_api_key_here":
                            os.environ[key] = value
            return


_load_env()


def get_llm_config() -> dict:
    """
    获取 LLM 配置字典。兼容新旧变量名。

    Returns:
        dict: {"api_key": str, "base_url": str, "model": str}

    Raises:
        ValueError: API Key 未设置
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "LLM_API_KEY 未设置。请在 model1.1/.env 文件中填入你的 API Key。"
        )

    return {
        "api_key": api_key,
        "base_url": os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", ""),
        "model": os.environ.get("MODEL_ID") or os.environ.get("DEEPSEEK_MODEL", ""),
    }


# ---------------------------------------------------------------------------
# LangChain ChatOpenAI 适配
# ---------------------------------------------------------------------------

def get_chat_model(temperature: float = 0.0):
    """
    返回适配 LLM 服务的 LangChain ChatOpenAI 实例。

    供 langgraph.prebuilt.create_react_agent 的 model 参数使用，
    也可直接调用 .invoke() 进行单次 LLM 请求。
    """
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=temperature,
    )
