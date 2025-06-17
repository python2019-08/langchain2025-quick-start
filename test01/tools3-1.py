import datetime
from langchain.tools import tool

#定义工具注意要添加注释
@tool
def get_current_date():
    """获取今天日期."""
    return datetime.datetime.today().strftime("%Y-%m-%d")
#大模型绑定工具
llm_with_tools = llm.bind_tools([get_current_date])
#工具容器
all_tools ={"get_current_date":get_current_date}
#把所有消息存到一起
query= "今天是几月几号" 
messages = [query]
#询问大模型。大模型会判断需要调用工具，并返回一个工具调用请求
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
#打印需要调用的工具
print(ai_msg.tool_calls)
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        selected_tool = all_tools[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
llm_with_tools.invoke(messages).content