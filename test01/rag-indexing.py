from langchain_community.document_loaders import TextLoader
#1、加载原始文档
loader = TextLoader("./resource/meituan-questions.txt")
documents = loader.load()
#2、切分文档
import re
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0 ,separator="\n\n",keep_separator=True)

texts = re.split(r"\n\n", documents[0].page_content)
segments = text_splitter.split_text(documents[0].page_content)

segment_documents = text_splitter.create_documents(texts)
#3、将文档向量化，保存到Redis中
import os
from langchain_community.embeddings import DashScopeEmbeddings
from config.load_key import load_key

if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("BAILIAN_API_KEY")
embedding_model = DashScopeEmbeddings(model="text-embedding-v1") ## alibailian

redis_url = "redis://localhost:6379"
from langchain_redis import RedisConfig,RedisVectorStore
config = RedisConfig(
    index_name="meituan-index",
    redis_url=redis_url
)

vector_store = RedisVectorStore(embedding_model,config=config)
vector_store.add_documents(segment_documents)