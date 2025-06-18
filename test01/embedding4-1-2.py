from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#定义两个文本
text1="我喜欢吃苹果"
text2="我最爱吃的水果是苹果"
text3="今天天气不错"
 
#获取文本向量
vector1 = np.array(embedding_model.embed_query(text1)).reshape(1,-1)
vector2 = np.array(embedding_model.embed_query(text2)).reshape(1,-1)
vector3 = np.array(embedding_model.embed_query(text3)).reshape(1,-1)
#计算余弦相似度
similarity12 = cosine_similarity(vector1, vector2)[0][0]
similarity13 = cosine_similarity(vector1, vector3)[0][0]
print(f"\"{text1}\" 与 \"{text2}\" 相似度: {similarity12:.4f}")
print(f"\"{text1}\" 与 \"{text3}\" 相似度: {similarity13:.4f}")