from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.alixuncs.com/compatible-mode/v1",
    openai_api_key=load_key("BAILIAN_API_KEY") ,
)
llm.invoke("如何退款？")