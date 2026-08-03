import pytest
from src.memory.history import ConversationHistory


@pytest.fixture
def history():
    return ConversationHistory()


def test_history_add_user(history):
    history.add_user("你是谁？")
    assert history.messages[0]["role"] == "user"
    assert history.messages[0]["content"]=="你是谁？"

def test_history_add_assistant(history):
    history.add_assistant("用户为小基")
    assert history.messages[0]["role"]=="assistant"
    assert history.messages[0]["content"]=="用户为小基"

def test_history_format(history):
    history.add_user("你是谁？")
    history.add_assistant("用户为小基")
    format_result=history.format()
    assert format_result =="用户: 你是谁？\n助手: 用户为小基"

def test_history_clear(history):
    history.add_user("你是谁？")
    format_result=history.format()
    assert len(history.messages)==1
    history.clear()
    assert len(history.messages)==0

def test_history_format_len(history):
    history.max_turns=1
    history.add_user("你是谁？我是小帅")
    history.add_assistant("用户为小基")
    history.add_user("你还记得我是小帅吗")
    format_result=history.format()
    assert format_result =="助手: 用户为小基\n用户: 你还记得我是小帅吗"
