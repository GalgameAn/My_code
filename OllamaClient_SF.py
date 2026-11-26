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
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
            f.flush()               # 把数据从内存刷到操作系统的缓冲区
            os.fsync(f.fileno())    # 强制操作系统把缓冲区的内容立刻写到物理硬盘上
        os.replace(tmp, path)       # 原子替换：把老文件干掉，把临时文件名改成正式文件名


    def restore(self, path):
        """从备份文件恢复对话历史"""
        self.load(path)
        if path is not None:    # 判断 path 是否为空，防止写坏 history_SF.json
            self.save()    
        print(f"索蕾娜：已从 {path} 中恢复记忆")        


    def load(self, path="history_SF.json"):
        """从本地文件反序列化，恢复对话历史到内存(带防崩溃保护)"""
        if  os.path.exists(path):  # 确认文件存在，然后安全读取
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
            except json.JSONDecodeError:
                # 如果 JSON 解析失败(文件坏了)
                corrupt = path + ".corrupt"     # 添加 .corrupt 后缀
                os.replace(path, corrupt)       # 把损坏文件改名备份
                print(f"索蕾娜：历史文件损坏，已备份到{corrupt}，开启新的文本框")
                self.messages = []
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

        # 确保 path 的指向目录存在，如 backup/
        parent = os.path.dirname(path)      # 提取出 backup 文件夹路径
        if parent:                          # 如果路径里未包含该文件夹
            os.makedirs(parent, exist_ok=True)      # 自动创建该文件夹

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

        url = f"{self.url}/api/chat"

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

        # 本轮暂存区，只记录每一轮的新消息(核心用于写入正式历史)
        turn_messages = [{"role":"user", "content":message_to_model}]

        # 发给 API 的完整上下文 = 历史 + 每轮暂存区
        api_messages = self.messages + turn_messages

        
        # 调用工具最大次数
        max_tool_calls = 4
        current_tool_calls = 0
        
        # 记录网络重试次数和最大次数
        max_retries = 3
        current_retries = 0
        

        # 开始计时
        start = time.time()

        print("索蕾娜：")
        while True:
            payload = {
                "model": self.model,
                "messages": api_messages,  
                "stream": False,                # 告知 Ollama 要非流式输出
                "options": {"num_ctx": 8192},   # 限制上下文窗口
                "tools": Tools.TOOLS,           #调用工具
                }
             
            try:
                reply = ""
                tool_calls = []

                with requests.post(url, json=payload, stream=False, timeout=(5,120)) as r:    
                    r.raise_for_status()    #抛出4、5类异常给HTTPError
                    data = r.json()

                    if "error" in data:
                        raise ValueError(f"Ollama返回错误：{data['error']}")   # 跳出所有循环，抛出 Ollama 的返回错误

                    data_message = data.get("message", {})
                    if data_message.get("content", ""):
                        reply = data_message.get("content", "")

                    tool_calls = data_message.get("tool_calls", "")                    
                    if tool_calls:
                        current_tool_calls += 1
                        if current_tool_calls > max_tool_calls:
                            print("[警告]调用工具次数过多，已强制终止")
                            turn_messages.append({"role": "assistant", "content": reply or "工具调用次数过多，无法继续处理。"})

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
                        turn_messages.append(assistant_msg)

                        # 把所有工具结果都追加到消息历史
                        for tr in tool_results:
                            turn_messages.append(tr) 

                        # 每次追加完，更新发给 API 的上下文
                        api_messages = self.messages + turn_messages

                        continue

                    else:
                        # 模型不打算调用工具了，直接回复了
                        if not reply:
                            reply = "...嗯。这种话题，我无法给予回答"
                            print(reply)

                        turn_messages.append({"role":"assistant", "content":reply})

                        break

            # 网络类错误：可以重试，但限制次数
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                current_retries += 1
                if current_retries > max_retries:
                    print(f"网络或 Ollama 连接失败已达{max_retries}次，强制终止本次对话。")
                    raise e     # 彻底抛出异常，让 Talk.py 去接住它，不在死循环

                print(f"连接失败，正在第{current_retries}次重试，将再次重试{max_retries - current_retries}次，请确保 Ollama 正常运行...")
                time.sleep(2)
                continue

            # HTTP 4xx 错误(譬如格式不对)和解析错误：重试无用，直接报错
            except requests.exceptions.HTTPError as e:
                print(f"HTTP 错误，可能是模型或代码格式问题，不再重试：{e}")
                raise e
            except (ValueError, KeyError, TypeError) as e:
                print(f"数据解析或响应错误：{e}")
                raise e

        # 存储我的原文本
        self.messages.append({"role":"user", "content":original_message})

        # 提取暂存区信息，并存入正式历史
        for msg in turn_messages[1:]:
            self.messages.append(msg)

        # 保存到硬盘和 RAG向量库
        self.save()
        self.auto_save_to_RAG(user_message=original_message, assistant_reply=reply)

        cost = time.time() - start
        if self.verbose:
            print(f"模型：{self.model}")
            print(f"耗时：{cost:.2f}s")