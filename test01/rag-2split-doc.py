from langchain_text_splitters import CharacterTextSplitter

# 切分文档
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0 ,separator="\n\n",keep_separator=True)

segments =text_splitter.split_documents(documents)
print(len(segments))
for segment in segments:
    print(segment.page_content)
    print("--------")