"""
huatai-agent — 华泰证券智能问数原型系统

ReAct 架构：LLM 自主探索数据库结构 → 生成 SQL → 执行验证 → 修正 → 输出答案。
基于 LangGraph create_react_agent，替代原来的 3 张子图固定流水线。

model1.2：单 Agent + Schema 缓存 + 多行粘贴输入
"""

import os
import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from tools.db_connector import _init_database
from dataclasses import dataclass

from agent.react_agent import build_agent


# ---------------------------------------------------------------------------
# 启动初始化
# ---------------------------------------------------------------------------

print("[INIT] 正在初始化数据库...")
_init_database()
print("[INIT] 数据库就绪")

print("[INIT] 正在构建 ReAct Agent...")
_agent = build_agent()
print("[INIT] Agent 就绪（Schema 缓存已启用）")


# ---------------------------------------------------------------------------
# 流式事件解析
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """从 LangGraph 流式消息中提取的结构化信息。

    一个 AgentStep 对应一次 LLM 回复——要么是思考+调工具，要么是最终回复。
    """
    content: str
    tool_calls: list
    is_thinking: bool   # True = 思考阶段（会调用工具）
    is_final: bool      # True = 最终回复（不再调工具）

    @classmethod
    def from_langgraph_msg(cls, msg):
        has_calls = bool(getattr(msg, "tool_calls", None))#取出message中的tool_call
        return cls(
            content=getattr(msg, "content", "") or "",
            tool_calls=getattr(msg, "tool_calls", []) or [],
            is_thinking=has_calls,
            is_final=not has_calls,
        )


def _iter_agent_steps(agent, user_input: str, config: dict):
    """把 Agent 的一次执行变成一连串清晰的"步骤"。

    目的：LangGraph 的 stream() 输出的是一个嵌套很深的原始事件流——
    event → node_name → update → messages[0] ——调用方要写 4 层缩进
    还要自己判断哪些是思考、哪些是最终回复。这个函数把这些封在这里，
    对外只暴露一个简单的迭代器：每次 yield 一个 AgentStep，
    让调用方只看到"这一步是思考"或"这一步是最终回复"。

    """
    # agent.stream 开始流式执行，每次 yield 一个事件（一个"步骤"）
    # stream_mode="updates" 表示只返回增量更新，不返回完整状态
    for event in agent.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="updates",
    ):
        # 每个 event 是一个 dict，key 是节点名（"agent" 或 "tools"），value 是输出
        for node_name, update in event.items():
            # 只关心 agent 节点（LLM 的思考/回复），不关心 tools 节点（工具执行结果）
            if node_name != "agent" or not update:
                continue
            # update["messages"] 是该节点的输出消息列表，取第一条
            # [None] 是兜底：如果 messages 为空，msg = None 而不是 IndexError
            msg = update.get("messages", [None])[0]
            # msg 可能为 None，也可能 content 为空字符串，都跳过
            if msg and getattr(msg, "content", None):
                # 把 LangGraph 原始消息转成 AgentStep，调用方不用关心内部结构
                yield AgentStep.from_langgraph_msg(msg)


# ---------------------------------------------------------------------------
# 多行输入（prompt_toolkit）。目的是，粘贴部分sql时会因为换行而直接发出去
# ---------------------------------------------------------------------------

def _create_input_session() -> PromptSession | None:
    """创建 prompt_toolkit 输入会话。非 TTY 环境返回 None。"""
    try:
        if not os.isatty(0):
            return None

        bindings = KeyBindings()

        @bindings.add('c-enter')
        def _send(event):
            """Ctrl+Enter 发送"""
            buf = event.current_buffer
            if buf.text.strip():
                buf.validate_and_handle()

        @bindings.add('enter')
        @bindings.add('s-enter')
        @bindings.add('c-j')
        def _newline(event):
            """Enter / Shift+Enter / Ctrl+J 换行"""
            event.current_buffer.insert_text('\n')

        @bindings.add('escape')
        def _clear(event):
            event.current_buffer.text = ''

        history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.huatai_history'
        )

        style = Style.from_dict({
            'prompt':     'ansigreen bold',
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#0066cc #ffffff',
        })

        return PromptSession(
            multiline=True,
            key_bindings=bindings,
            history=FileHistory(history_path),
            style=style,
        )

    except Exception as e:
        print(f"  [WARN] 无法创建高级输入模式: {e}")
        return None


def _fallback_input() -> str | None:
    """简化输入：Enter 换行，空行提交。EOF 时返回 None。"""
    lines = []
    print()
    while True:
        prompt = ">>> " if not lines else "... "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        if line.strip():
            lines.append(line)
        elif lines:
            break  # 空行提交
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 查询执行
# ---------------------------------------------------------------------------

def run_query(user_input: str, thread_id: str = None):
    """
    执行一次查询，流式输出推理过程，返回 Agent 的最终回复文本。

    Args:
        user_input: 用户自然语言输入
        thread_id:  会话 ID（用于多轮对话记忆）

    Returns:
        str: Agent 的最终回复
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    final_content = ""
    first_thought = True

    for step in _iter_agent_steps(_agent, user_input, config):
        if step.is_thinking:
            # 显示思考过程
            thought = step.content
            if thought:
                if first_thought:
                    print(f"\n  ╔══════════════╗")
                    print(f"  ║  容我三思   ║")
                    print(f"  ╚══════════════╝")
                    first_thought = False
                print(f"  [思考] {thought}")

        elif step.is_final:
            final_content = step.content

    return final_content or "（Agent 未产生回复）"


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  huatai-agent 1.2  单 Agent 取数系统")
    print("  特性: Schema 缓存 + 多行粘贴")
    print("=" * 60)

    # ---- 创建输入会话 ----
    session = _create_input_session()

    if session is not None:
        print("  Ctrl+Enter 发送 | Enter/Shift+Enter 换行 | ↑↓ 历史 | Esc 清空")
    else:
        print("  [简化模式] Enter 换行 | 空行提交")

    print("  quit 退出")
    print("  示例问题:")
    print("    学历本科及以上的男性客户，年龄大于50岁的有多少个")
    print("    不同客户年龄段资产分布情况")
    print("    查询资产减值最多的客户")
    print("=" * 60)

    thread_id = str(uuid.uuid4())

    while True:
        try:
            if session is not None:
                user_input = session.prompt(
                    HTML('\n<ansigreen><b>&gt; </b></ansigreen>')
                )
            else:
                user_input = _fallback_input()
                if user_input is None:
                    print("\n再见")
                    break

        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        user_input = user_input.strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见")
            break

        if not user_input:
            continue

        print()
        response = run_query(user_input, thread_id=thread_id)
        print(response)
        print("-" * 60)
