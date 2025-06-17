
from langchain_core.chat_history import InMemoryChatMessageHistory
# 这是BaseChatMessageHistory的子类
history = InMemoryChatMessageHistory()
#第一轮聊天
history.add_user_message("你是谁?")
aimessage = llm.invoke(history.messages)
print(aimessage.content)
history.add_message(aimessage)
#第二轮聊天如果没有上一次聊天的记录，大模型是不知道要重复什么内容的。
history.add_message("请重复一次")
aimessage2 = llm.invoke(history.messages)
print(aimessage2.content)
history.add_message(aimessage2)
#打印历史聊天记录
# print("Chat History:")
# for message in history.messages:
#   print(f"{type(message).__name__}: {message.content}")