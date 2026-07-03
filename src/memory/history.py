"""
对话历史管理
"""
from src.config import config


class ConversationHistory:
    """管理多轮对话历史，自动截断到最近 N 轮"""

    def __init__(self, max_turns: int | None = None):
        self.messages: list[dict] = []
        self.max_turns = max_turns or config.HISTORY_MAX_TURNS

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def format(self) -> str:
        """格式化为 Prompt 可用的文本，只保留最近 max_turns 轮"""
        recent = self.messages[-(self.max_turns * 2):]  # ×2 因为每轮 = user + assistant
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self.messages.clear()
