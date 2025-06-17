from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate

# LCEL定制一个chain
prompt = ChatPromptTemplate.from_messages([("human","你好，请用下面这种语言回答我的问题 {language}.")])

parser = StrOutputParser()

chain = prompt | llm | parser

# 将chain转换成工具
as_tool =chain.as_tool(name="translatetool",description="翻译任务")

all_tools = {"translatetool":as_tool}

print(as_tool.args)
# 绑定工具
llm_with_tools = Llm.bind_tools([as_tool])

query= "今天天气真冷，这句话用英语怎么回答？" 
messages = [query]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
print(ai_msg.tool_calls)
print(">>>>>>>>>>>>")
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        selected_tool = all_tools[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
llm_with_tools.invoke(messages).content