from langchain_core.prompts import ChatPromptTemplate

#创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ])
#格式转换函数，prompt.invoke方法返回PromptValue，而retriver.invoke需要传入的参数为str。中间做个格式转换
def format_prompt_value(prompt_value) :
    return prompt_value.to_string()

#链式连接检索器和提示模板
chain = prompt | format_prompt_value | retriver
#调用链并传入用户的问题
documents=chain.invoke({"question":"又长又甜的水果是什么？"})
for document in documents:
    print(document.page_content)