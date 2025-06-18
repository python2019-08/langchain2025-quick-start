import os
from config.load_key import load_key

if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")

from langchain_community.embeddings import DashScopeEmbeddings
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

text = "This is a test query."
query_result = embedding_model.embed_query(text)
print(query_result) #打印向量结果
print(len(query_result)) # 向量维度