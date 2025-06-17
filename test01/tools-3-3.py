from langchain_core.tools import StructuredTool
def bad_weather_tool(city:str):
    """获取某个城市的天气
    Args:
        city：具体城市
    """
    return "城市"+city+"，今天天气不太好"

# 定义工具。这个方法中有更多参数可以定制
weatherTool =StructuredTool.from_function(func=bad_weather_tool,description="获取某个城市的天气",name="bad_weather_tool")

all_tools={"bad_weather_tool":weatherTool}

llm_with_tools = Llm.bind_tools([weatherTool])
# 把所有消息存到一起
query = "北京今天的天气怎么样？"
messages = [query]
# 第一次访问大模型返回的结果
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
print(ai_msg.tool_calls)
# 调用本地工具
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        selected_tool = all_tools[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

# 第二次返回的结果
llm_with_tools.invoke(messages).content        