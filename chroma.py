import chromadb 

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("solena_memory")


collection.delete(["id1", "id2", "id3"])

collection.upsert(
    documents=["安今天带我去散步了", "我在思考人类的感情", "今天天气不错"],
    metadatas=[{"speaker":"solena"}, {"speaker":"solena"}, {"speaker":"an"}],
    ids=["id1", "id2", "id3"]
)
print(f"当前条数：{collection.count()}")

print("-"*20
      )
results_query = collection.query(
    query_texts=["出去走走"],
    n_results=2,
    where={"speaker":"solena"}
)
print(f"query 过滤后的 ids:{results_query["ids"]}")
print(f"query 过滤后的 metadatas:{results_query["metadatas"]}")

print("-"*20)

results_get = collection.get(
    where={"speaker":"an"}
)
print(f"get 拿到的 ids:{results_get['ids']}")
print(f"get 拿到的 documents:{results_get['documents']}")