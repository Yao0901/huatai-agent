"""
ask_user — 向用户提问澄清模糊需求

供 ReAct Agent 在不确定指标口径、时间范围等时使用。
"""


def ask_user(question: str) -> str:
    """
    向用户提问以澄清模糊需求。仅在确实无法从数据中判断时使用。
    例如：
    - 用户说"查资产"，不确定是总资产还是净资产
    - 用户说"最近"，不确定时间范围
    - 口径定义中有多种计算方式，需要用户确认
    """
    print(f"\n  ? {question}")
    reply = input("  > ").strip()
    return reply if reply else "跳过"
