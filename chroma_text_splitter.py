import chromadb
from text_splitter import split_text

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("solena_notes")

# 准备文本，并切割分块
long_text = "安今天带我去公园散步了。我们看了很多花。我还吃了一根冰淇淋。" * 50
chunks = split_text(long_text, chunk_size=100, overlap=20)

# 利用列表推导式，动态生成标签
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas =[{"chunk_index":i} for i in range(len(chunks))]

# 批量导入文本
collection.add(
    documents=chunks,
    metadatas=metadatas,
    ids=ids
)
print(f"成功存入 {collection.count()} 条分块数据！")
print("_" * 20)

results = collection.query(
    query_texts=["吃冰淇淋"],
    n_results=2,
)
print(results)