from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnableWithMessageHistory

#提示词模板
prompt_template_zh = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from English into Chinese"),
        ("user", "{text}")
        ])

prompt_template_fr = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from English into French"),
        ("user","{text}")
        ])

#构建链
chain_zh = prompt_template_zh | llm | parser
chain_fr = prompt_template_fr | llm | parser

#并行执行两个链
parallel_chains = RunnableMap({
    "zh_translation": chain_zh,
    "fr_translation": chain_fr
    })

# 合并结果
final_chain = parallel_chains | RunnableLambda(lambda x: f"Chinese: {x['zh_translation']}\nFrench: {x['fr_translation']}")

# 调用链
print(final_chain.invoke({"text":"nice to meet you"}))