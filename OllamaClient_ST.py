import time
import os
import json
import requests
from datetime import datetime
from solena_System_Prompt import SYSTEM_PROMPT


def get_current_time():
    """报时"""
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #datetime.now()>>获取当前时间  strftime("%Y-%m-%d %H:%M:%S")>>转换 年-月-日 时-分-秒 的格式


def calculator(operation, a, b):
    """计算器函数：支持加减乘除"""
    operation = operation.strip().lower()
    a = float(a)
    b = float(b)
    if operation == '+':
        return str(a + b)
    elif operation == '-':
        return str(a - b)
    elif operation == '×' or operation == '*':
        return str(a * b)
    elif operation == '÷' or operation == '/':
        if b == 0:
            return "错误：除数不能为零"
        return str(a / b)
    else:
        return f"错误：不支持的操作 '{operation}'，仅支持 + - × ÷"
    

def get_weather(city='武汉'):
    """查询指定城市的天气"""
    url = "https://wttr.in/" + city + "?format=3&lang=zh"
    try:
        r = requests.get(url, timeout=300)
        return r.text.strip()
    except:
        return f"暂时查不到{city}的天气"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，当用户询问现在几点、今天几号时使用",
            "parameters": {     
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持加减乘除。当用户需要计算数学表达式时使用",
            "parameters": {     # 创建一个参数包
                "type": "object",   # 让参数按照标准格式（键值对）排序的一个命令
                "properties": {     # 包含参数具体内容de'bao
                    'operation':{   # 参数名称
                        'type':'string',    # 参数类型
                        'description':'运算符，可选值：+（加）、-（减）、×（乘）、÷（除）'
                        },
                    'a':{
                        'type':'number',
                        'description':'第一个数'                         
                        },
                    'b':{
                        'type':'number',
                        'description':'第二个数',
                        }
                    },
                "required": ['operation', 'a', 'b']   #["必须提供的参数列表"]
            }
        }
    },
    {
        'type':'function',
        'function':{
            'name':'get_weather',
            'description':'查询指定城市天气',
            'parameters':{
                'type':'object',
                'properties':{
                    'city':{
                        'type':'string',
                        'description':'城市名称，如武汉、北京',
                    }
                },
                'required':['city']
            }
        }
    }
]


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
        self.messages = []      # 多轮对话-记忆库

    def save(self, path="history.json"):
        """把对话历史序列化后写进本地文件，相当于给记忆拍快照"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def backup(self, path=None):
        """备份当前对话历史，默认生成带时间戳的文件名"""
        if path is None:
            # 没传文件名，自动生成带时间戳的
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"backup_{ts}.json"

        with open(path, 'w',encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False,indent=2)

        print(f"solena：已备份到 {path}")

    def restore(self, path):
        """从备份文件恢复对话历史"""
        self.load(path)
        if path is not None:    # 判断 path 是否为空，防止写坏 history.json
            self.save()    
        print(f"solena：已从 {path} 中恢复记忆")        

    def load(self, path="history.json"):
        """从本地文件反序列化，恢复对话历史到内存"""
        if  os.path.exists(path):  # 确认文件存在，然后安全读取
            with open(path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
        else:
            print("solena：历史文本不存在，开启新的文本框")
            self.messages = []

        # 防止 system 重复出现，确保人格稳定性
        if not any(m.get("role") == "system" for m in self.messages):
            self.messages.insert(0, {'role':'system', 'content':SYSTEM_PROMPT})

    def chat(self, message: str) -> str:
        """发送用户消息，拿到回复后存盘"""
        self.messages.append({"role": "user", "content": message})  # 追加我的文本
        url = f"{self.url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.messages,  # 投喂记忆库-多轮对话的核心
            "stream": True,  # 告知 Ollama 要流式输出
            "options": {"num_ctx": 2048},  # 限制上下文窗口
            "tools": TOOLS,     #调用工具
        }

        try:
            full_reply = ""
            start = time.time()  # 开始计时
            tool_calls = None
            print("solena：",end="")
            with requests.post(url, json=payload, stream=True) as r:    # 向 Ollama 发出第一次发送请求，可能输出[日常回复信息(self.messages)]，或存储[调用工具结果(TOOLS)]并发送二次请求
                r.raise_for_status()    #抛出4、5类异常给HTTPError
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)                      
                        data = chunk.get("message",{})

                        # 在循环内捕获含tool_calls的消息
                        if data.get("tool_calls"):      # {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}]}
                            tool_calls = data.get("tool_calls")
                            tool_calls_message = data

                        content = data.get("content","")
                        if content:
                            full_reply += content
                            print(content, end='', flush=True)

                if tool_calls:
                    # 支持多个工具
                    tool_results = []
                    for tool_call in tool_calls:

                        # 拿到模型想调的函数名和参数
                        func = tool_call.get('function', {})
                        name = func.get('name', '')
                        args = func.get('arguments', '')

                        # 如果 args 是字符串，json.loads 转字典
                        if isinstance(args, str):
                            args = json.loads(args)


                        # 执行对应的函数
                        if name == 'get_current_time':
                            result = get_current_time()
                        elif name == "calculator":
                        # calculator 需要 operation, a, b 三个参数
                            result = calculator(
                                operation=args.get('operation'),
                                a=args.get('a'),
                                b=args.get('b'),
                            )
                        elif name == 'get_weather':
                            result = get_weather(city=args.get('city', '武汉'))
                        else:
                            result = f"不知道这个函数: {name}"


                        # 追加所有工具结果
                        tool_results.append({
                            'role':'tool',
                            'content':result,
                            'tool_call_id': tool_call.get('id', f"call_{len(tool_results)}")     # [1]模型返回的每个工具调用都有一个唯一 ID，比如 "call_9xpdxqaf"。[2]"id" 有时不一定存在，默认值 f'call_{len(tool_results)}'会返回有序 ID（比如 'call_0'、'call_1'）>>> 原理：用叠加的数量来代表顺序
                        })

                    # 让模型知道它调了工具
                    self.messages.append(tool_calls_message)

                    # 把所有工具结果都追加到消息历史
                    for tr in tool_results:
                        self.messages.append(tr)

                    payload2 = {
                        'model':self.model,
                        'messages':self.messages,
                        'stream':False,
                        }

                    r2 = requests.post(url, json=payload2, timeout=300)     # 向 Ollama 发出第二次请求,输出[工具结果(经过包装)]
                    r2.raise_for_status()
                    data2 = r2.json()

                    # 防止模型陷入调用工具的死循环
                    message2 = data2.get('message', {})
                    if message2.get('tool_call'):
                        raise ValueError("第二次请求又收到了tool_calls，模型可能无法正确处理")  #raise 的作用当程序运行到 raise 这一行时，立刻中断当前流程，把异常信息一层一层往外抛，直到被 try-except 捕获，或者程序直接崩溃并打印错误信息。示例：流程：raise ValueError(...) 被触发程序立刻跳出 try 块，进入 except 块except 捕获到这个 ValueError，把它存到变量 e 里打印错误信息，返回错误字符串
                    reply2 = message2.get('content', '')

                    
                    # 把最终回复也存入历史并保存
                    self.messages.append({"role": "assistant", "content": reply2})
                    self.save()

                    # 输出调用工具结果
                    print(reply2)
                
                # 确保真正的错误信息被暴露出来
                if "error" in data:
                    print(f"请求失败：{data['error']}") 
                
                self.messages.append(
                    {"role": "assistant", "content": full_reply}
                )  # 追加模型回答
                
                cost = time.time() - start  # 结束计时，记录请求用时

                if self.verbose:
                    print(f"模型：{self.model}")
                    print(f"耗时：{cost:.2f}s")

                self.save()  # 确保在程序崩溃时，减小文本损失

        except requests.exceptions.ConnectionError:
            print("请求失败：请确保ollama是否正常运行中") 
        except requests.exceptions.Timeout:
            print("请求失败：连接超时，请稍后重试") 
        except requests.exceptions.HTTPError as e:
            # 打印Ollama实际返回的内容，方便排查
            print(f"请求失败：HTTP错误 {e}") 
        except (ValueError, KeyError, TypeError) as e:
            # 打印Ollama实际返回的内容，方便排查
            print(f"请求失败：{e}")

    def clear_history(self):
        """清除记忆库"""
        self.messages = []
        self.save()  # 把空列表写回去，覆盖旧文件
        self.messages.insert(0, {'role':'system', 'content':SYSTEM_PROMPT})     # 确保 clear 后，人格的稳定性
        # self.save() 被注释掉原因：[1]同一对话框内，chat 方法调用过 save 方法  [2]程序崩溃时，必定重启，届时又有 load 方法中的判断防御
                        