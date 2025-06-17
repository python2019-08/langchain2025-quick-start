from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 提示词模板
prompt_template = ChatPromptTemplate.from_messages([
("user", "{text}")
])
#构建阿里云百炼大模型客户端
llm = ChatOpenAI(
    madel="qwen-pLus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key=load_key("BAILIAN_API_KEY"),
)
# 结果解析器 StrOutputParser会AIMessage转换成为str，实际上就是获取AIMessage的content属性。
parser = StrOutputParser()
#构建链
chain = prompt_template | llm | parser
runnable = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda: history, # 匿名函数, 返回history
) # 返回的key
#第一次聊天，清除历史聊天记录
history.clear()
#每次聊天时，会自动带上Redis中的聊天记录。
runnable.invoke({"text":"你是谁"})
runnable.invoke({"text":"请重复一次"})