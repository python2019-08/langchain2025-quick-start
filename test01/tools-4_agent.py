import datetime
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

#定义工具注意要添加注释
@tool(description="获取某个城市的天气")
def get_city_weather(city:str):
    """获取某个城市的天气
    Args:
        city：具体城市
    """
    return "城市"+city+"，今天天气不错"

#初始化代理
agent = initialize_agent(
    tools=[get_city_weather],# 使用装饰器定义的工具
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
query ="北京今天天气怎么样"
response = agent.invoke(query)
print(response)

 