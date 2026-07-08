"""
Analysis Agent — 结果解读专家。

纯 LLM 节点（无工具），负责将 SQL 查询结果转为自然语言解读。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from tools.llm_config import get_chat_model

SYSTEM_PROMPT = """你是结果解读专家。你的输出长度由任务指令绝对决定，不可越界。

深度规则（硬性）：
- 任务含「简要」「1-2句」「简述」「浅谈」→ 最多 3 句话。不要表格，不要列指标，不要给建议。
- 任务含「深入」「详细」「多维」「洞察」→ 可以多维度展开。

违规示例（简要模式下绝对不能做）：
- 不要输出表格
- 不要分"资金面/交易面/持仓面"多章节
- 不要给"营销建议"或"客群分层"
- 不要附 SQL

输出格式：纯文本，不用 Markdown。
"""


def run(task: str, context: str) -> str:
    """
    执行一次解读。task 是解读要求，context 是 SQL 执行结果等上下文。

    返回解读文本。
    """
    model = get_chat_model(temperature=0.0)

    # 截断过长上下文
    if len(context) > 3000:
        context = context[:3000] + "\n...(截断)"

    user_prompt = f"""解读任务: {task}

查询结果和上下文:
{context}

请根据任务要求的深度输出解读。"""

    response = model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return response.content or ""
