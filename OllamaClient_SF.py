import time, uuid, chromadb, os, json, requests, Tools
from datetime import datetime
from Privacy import desensitize
from solena_System_Prompt import SYSTEM_PROMPT
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


class OllamaClient:
    """调用本地ollama模型"""

    def __init__(
        self,
        base_url="http://localhost:11434",
        model="qwen3.5:4b-q4_K_M",
        verbose=False,   # 计时开关
    ):
        self.url = base_url.strip("/")
        self.model = model
        self.verbose = verbose

        # 多轮对话-短期记忆库
        self.messages = []

        # 多轮对话-长期记忆库
        self.ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434", model_name="bge-m3")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="solena_memory", embedding_function=self.ollama_ef)


    def save(self, path="history_SF.json"):
        """把对话历史序列化后写进本地文件，相当于给记忆拍快照"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)


    def restore(self, path):
        """从备份文件恢复对话历史"""
        self.load(path)
        if path is not None:    # 判断 path 是否为空，防止写坏 history_SF.json
            self.save()    
        print(f"索蕾娜：已从 {path} 中恢复记忆")        


    def load(self, path="history_SF.json"):
        """从本地文件反序列化，恢复对话历史到内存"""
        if  os.path.exists(path):  # 确认文件存在，然后安全读取
            with open(path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
        else:
            print("索蕾娜：历史文本不存在，开启新的文本框")
            self.messages = []

        # 防止 system 重复出现，确保人格稳定性
        if not any(m.get("role") == "system" for m in self.messages):
            self.messages.insert(0, {'role':'system', 'content':SYSTEM_PROMPT})


    def clear_history(self):
        """清除记忆库"""
        # 把空列表写回去，覆盖旧文件
        self.messages = []
        self.save() 
        
        # 确保 clear 后，人格的稳定性
        self.messages.insert(0, {'role':'system', 'content':SYSTEM_PROMPT})
        # self.save() 被注释掉原因：[1]同一对话框内，chat 方法调用过 save 方法  [2]程序崩溃时，必定重启，届时又有 load 方法中的判断防御
    
    
    def backup(self, path=None):
        """备份当前对话历史，默认生成带时间戳的文件名"""
        if path is None:
        # 没传文件名，自动生成带时间戳的
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"backup/backup_{ts}.json"
    
        with open(path, 'w',encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False,indent=2)
    
        print(f"索蕾娜：已备份到 {path}")
    

    def auto_save_to_RAG(self, user_message, assistant_reply):
        """自动把对话存储到长期记忆"""
        # 拼装文本
        raw_text = f"安：{user_message}\n索蕾娜：{assistant_reply}"

        # 文本脱敏
        safe_text = desensitize(raw_text)

        # 自动生成唯一随机 ID
        memory_ID = str(uuid.uuid4())

        # 存入 ChromaDB (未切块版)
        try:
            self.collection.upsert(documents=safe_text, ids=memory_ID)
        except Exception as e:
            print(f"长期记忆存储失败，详情：{e}")


    def chat(self, message):
        """发送用户消息，拿到回复后存盘"""

        # 检索长期记忆
        results = self.collection.query(query_texts=[message], n_results=2)
        retrieved_memories = results["documents"][0]
        retrieved_distances = results["distances"][0]

        # 过滤检索记忆
        context = ""
        valid_memories = []
        original_message = message      # 备份原文本，防止污染记忆库
        message_to_model = original_message     # 初始化变量，防止污染喂给变量的内容
        for i, r_d in enumerate(retrieved_distances):
            if r_d < 0.5:
                valid_memories.append(retrieved_memories[i])

        # 缝合检索文本
        if valid_memories:
            context = "\n".join(valid_memories)
            message_to_model = f"结合以下过去的回忆回答我：\n{context}\n\n用户现在提问：{original_message}"

        # 追加我的文本(注意录入原文本)
        self.messages.append({"role": "user", "content": original_message})  

        url = f"{self.url}/api/chat"
        max_tool_calls = 4
        current_tool_calls = 0

        # 开始计时
        start = time.time()

        print("索蕾娜：",end="")
        while True:
            # 浅拷贝——修改喂给服务端的内容
            message_to_api = self.messages.copy()
            if message_to_api[-1]["role"] == "user":
                message_to_api[-1] = {"role":"user", "content":message_to_model}
            
            payload = {
                "model": self.model,
                "messages": message_to_api,  # 投喂记忆库-多轮对话的核心
                "stream": False,  # 告知 Ollama 要非流式输出
                "options": {"num_ctx": 8192},  # 限制上下文窗口
                "tools": Tools.TOOLS,     #调用工具
                }
             
            try:
                reply = ""
                tool_calls = []

                with requests.post(url, json=payload, stream=False, timeout=(5,120)) as r:    
                    r.raise_for_status()    #抛出4、5类异常给HTTPError
                    data = r.json()

                    if "error" in data:
                        print(f"请求错误：{data['error']}")
                        break

                    data_message = data.get("message", {})
                    if data_message.get("content", ""):
                        reply = data_message.get("content", "")

                    tool_calls = data_message.get("tool_calls", "")                    
                    if tool_calls:
                        current_tool_calls += 1
                        if current_tool_calls > max_tool_calls:
                            print("[警告]调用工具次数过多，已强制终止")
                            self.messages.append({"role": "assistant", "content": reply or "工具调用次数过多，无法继续处理。"})
                            self.save()

                            break

                        # 多个工具
                        tool_results = []
                        for tool_call in tool_calls:

                            # 拿到模型想调的函数名和参数
                            func = tool_call.get('function', {})
                            name = func.get('name', '')
                            args = func.get('arguments', '')

                            # 如果 args 是字符串，json.loads 转字典
                            try:
                                if isinstance(args, str):
                                    args = json.loads(args)
                            except json.JSONDecodeError as e:
                                print(f"JSON 解析失败：{e}")
                                args = {}

                            # 执行对应的函数
                            if name == 'get_current_time':
                                result = Tools.get_current_time()
                            elif name == "calculator":
                                result = Tools.calculator(operation=args.get('operation'),a=args.get('a'), b=args.get('b'))   # calculator 需要 operation, a, b 三个参数
                            elif name == 'get_weather':
                                result = Tools.get_weather(city=args.get('city', '武汉'))
                            else:
                                result = f"不知道这个函数: {name}"

                            # 追加所有工具结果
                            tool_call_id = tool_call.get('id', f"call_{len(tool_results)}")   # [1]模型返回的每个工具调用都有一个唯一 ID，比如 "call_9xpdxqaf"。[2]"id" 有时不一定存在，默认值 f'call_{len(tool_results)}'会返回有序 ID（比如 'call_0'、'call_1'）>>> 原理：用叠加的数量来代表顺序
                            tool_results.append({'role':'tool', 'content':str(result), 'tool_call_id': tool_call_id})

                        # 记录模型调用工具的结果 reply 和过程 tool_calls_msg
                        assistant_msg = {'role':'assistant', 'content':reply, 'tool_calls':tool_calls}
                        self.messages.append(assistant_msg)

                        # 把所有工具结果都追加到消息历史
                        for tr in tool_results:
                            self.messages.append(tr) 
                        
                        self.save()

                        continue

                    else:
                        # 模型不打算调用工具了，直接回复了
                        
                        cost = time.time() - start  # 结束计时，记录请求用时                                                           
                        if self.verbose:
                            print(f"\n模型：{self.model}")
                            print(f"耗时：{cost:.2f}s")
                                               
                        if reply:      # 预防空回复污染历史文本
                            print(reply)
                            self.messages.append({"role": "assistant", "content": reply})  # 追加模型回答
                            self.save()
                            self.auto_save_to_RAG(user_message=original_message, assistant_reply=reply)    # 注意录入原文本
                        else:
                            # 告知模型，存在回答不了的问题时，响应方法，而不是空回复或崩溃
                            empty_reply = "……嗯。这种话题，我暂时不展开。"
                            print(empty_reply)
                            self.messages.append({"role": "assistant", "content": empty_reply})
                            self.save()
                     
                        break

            except requests.exceptions.ConnectionError:
                print("请求失败：请确保ollama是否正常运行中") 
                continue           
            except requests.exceptions.Timeout:
                print("请求失败：连接超时，请稍后重试") 
                continue
            except requests.exceptions.HTTPError as e:
                # 打印Ollama实际返回的内容，方便排查
                print(f"请求失败：HTTP错误 {e}")
                continue 
            except (ValueError, KeyError, TypeError) as e:
                # 打印Ollama实际返回的内容，方便排查
                print(f"请求失败：{e}")
                continue          
                        