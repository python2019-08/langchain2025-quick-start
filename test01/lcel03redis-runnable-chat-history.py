import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
from config.load_key import load_key


apiKey =load_key("ALiBaiLianDefault-api-key")
print(apiKey)


from langchain_redis import RedisChatMessageHistory
#这也是BaseChatMessageHistory的子类。本地需启动Redis服务
history = RedisChatMessageHistory(session_id="test",redis_url="redis://locaLhost:6379/0")
history.add_user_message("你是谁? ")
"""
```sh
% docker exec -it redis-stack redis-cli

127.0.0.1:6379> keys *
 1) "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"
 2) "runoobkey"
 3) "greeting"
 4) "chat:test:01JXY0F23ANZTF51PD4HZGHBYG"
 5) "runlist"
 6) "chat:test:01JXY0F9KD8NTFSX7W05FAWZHV"
 7) "runset"
 8) "runset2"
 9) "chat:test:01JXY6E0MRQAJ8H2WY1NET2ET0"
10) "chat:test:01JXY6E0MQS8WSTEYAFQHJ1X1C"
11) "chat:test:01JXY0F9KCEJYP8XP8C68C2Y9R"
127.0.0.1:6379> get "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"
(error) WRONGTYPE Operation against a key holding the wrong kind of value
127.0.0.1:6379> TYPE chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F
ReJSON-RL
127.0.0.1:6379> type "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"
ReJSON-RL
127.0.0.1:6379> JSON.GET "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"
"{\"type\":\"human\",\"message_id\":\"01JXY6DV3CVV6EQB5FAGJY3G2F\",\"data\":{\"content\":\"\xe4\xbd\xa0\xe6\x98\xaf\xe8\xb0\x81? \",\"additional_kwargs\":{},\"type\":\"human\"},\"session_id\":\"test\",\"timestamp\":1750138809.452972}"
127.0.0.1:6379> JSON.GET "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"  $.type
"[\"human\"]"
127.0.0.1:6379> JSON.GET "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"  $.data
"[{\"content\":\"\xe4\xbd\xa0\xe6\x98\xaf\xe8\xb0\x81? \",\"additional_kwargs\":{},\"type\":\"human\"}]"
127.0.0.1:6379> JSON.GET "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"  $.message_id
"[\"01JXY6DV3CVV6EQB5FAGJY3G2F\"]"
```
"""


from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
# 构建阿里云百炼大模型客户端
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apiKey,
)

runnable = RunnableWithMessageHistory(
    llm,
    get_session_history=lambda: history, # 匿名函数，返回history
)
# 返回的key
#第一次聊天清除历史聊天记录
#history.clear()
#runnable.invoke（{"text":"你是谁"})
#之后每次聊天时，会自动带上Redis中的聊天记录。
ret=runnable.invoke({"text":"请重复一次"})
print(ret)
