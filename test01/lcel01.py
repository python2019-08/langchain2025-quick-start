import os
import sys


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

now_dir = os.getcwd()
sys.path.append(now_dir)
from config.load_key import load_key
from langchain_openai import ChatOpenAI

apiKey =load_key("ALiBaiLianDefault-api-key")
print(apiKey)

# 提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system","Translate the following from English into {language}"),
    ("user","{text}")
    ]) 
# 构建阿里云百炼大模型客户端
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apiKey,
)

# 结果解析器 StrOutputParser会AIMessage转换成为str，实际上就是获取AIMessage的content属性
parser = StrOutputParser()

# 构建链 remark:用管道|串联
chain = prompt_template | llm | parser

# 直接调用链
print(chain.invoke({"text":"nice to meet you", "language":"chinese"}))

#继续构建更复杂的链
analysis_prompt=ChatPromptTemplate.from_template("我应该怎么回答这句话？{talk}。给我一个五个字以内的示例")
chain2 = {"talk":chain} | analysis_prompt | llm | parser
print(chain2.invoke({"text":"nice to meet you", "language":"Chinese"}))