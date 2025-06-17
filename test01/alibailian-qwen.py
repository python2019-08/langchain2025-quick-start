import os
import sys


# -----------------------
now_dir = os.getcwd()
sys.path.append(now_dir)

# ALiBaiLian-api-key is not ok
from config.load_key import load_key
apiKey =load_key("ALiBaiLianDefault-api-key")
print(apiKey)

 
if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = apiKey
               
# -----------------------------------------------------    
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages.human import HumanMessage

llm = ChatTongyi(
    model="qwen-plus",
    api_key=apiKey,
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
)
# ret=llm.invoke([HumanMessage("你是谁？你能帮我解决什么问题？")])       
ret=llm.invoke("你是谁？你能帮我解决什么问题？")  
print(ret)


#------------------------------------------------------------  
# https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.60907980kmpyaQ&tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html  
def code_from_bailian_console_aliyun():  
    from openai import OpenAI


    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    print(completion.model_dump_json())
#------------------------------------------------------------ 