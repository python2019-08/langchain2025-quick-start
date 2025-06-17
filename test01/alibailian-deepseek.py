import os
import sys

# -----------------------
now_dir = os.getcwd()
sys.path.append(now_dir)

from config.load_key import load_key
apiKey =load_key("ALiBaiLianDefault-api-key")
print(apiKey)

# -----------------------  
from langchain_openai import ChatOpenAI
from config.load_key import load_key
from langchain_core.messages.human import HumanMessage 

llm = ChatOpenAI(
    # model="deepseek-v3",
    model ="deepseek-r1",  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apiKey
    )

ret=llm.invoke([HumanMessage("你是谁？你能帮我解决什么问题？")])        
print(ret)