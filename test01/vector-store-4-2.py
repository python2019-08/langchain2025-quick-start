redis_url = "redis://localhost:6379"

import redis
redis_client = redis.from_url(redis_url)
print(redis_client.ping())# 测试连接返回True表示连接成功


from langchain_redis import RedisConfig,RedisVectorStore
config = RedisConfig(
    index_name="fruit",
    redis_url=redis_url
    )
vector_store = RedisVectorStore(embedding_model,config=config)
vector_store.add_texts(["香蕉很长","苹果很甜","西瓜又大又圆"])
scored_results = vector_store.similarity_search_with_score("又圆又大的水果是什么",k=3)
for doc, score in scored_results:
    print(f"{doc.page_content} - {score}")