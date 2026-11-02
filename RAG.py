import chromadb
import requests
import json 
from Text_splitter import split_text
from Privacy import desensitize
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434", model_name="bge-m3")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="solena_notes_v2", embedding_function=ollama_ef)
    
# collection.delete(ids=[f"chunk_{i}" for i in range(19)])

memories = [
    "安最喜欢的水果是苹果。",
    "安昨天带我去爬山，累坏了。",
    "安说他最近在学 Python 和 AI。",
    "今天天气是阴天，可能下雨。"
]

collection.upsert(documents=memories, ids=["m1", "m2", "m3", "m4"])

user_question = input("安：")

results = collection.query(query_texts=[user_question], n_results=2)
retrieved_memories = results["documents"][0]
retrieved_distances = results["distances"][0]
r_m = []

for i,r_d in enumerate(retrieved_distances):
    if r_d < 0.5:
        r_m.append(retrieved_memories[i])
        print(f"索蕾娜：已检索到符合的记忆：{retrieved_memories[i]}\n\t对应的距离为：{r_d}")

if r_m:
    context = "\n".join(r_m)
    system_prompt = f"你叫索蕾娜，是安的私人 AI。以下是你们之前的记忆：\n{context}\n请结合记忆回答安的问题，不要编造。"

    url = "http://localhost:11434/api/chat"
    payload = {
        "model":"qwen3.5:4b-q4_K_M",
        "messages":[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_question}],
        "stream":True,
        "options":{"num_ctx": 8192}
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)

else:
    print("索蕾娜：...不存在符合条件的记忆")