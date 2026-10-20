import time
import os
import json
import requests
from datetime import datetime


def get_current_time():
    """报时"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持加减乘除。当用户需要计算数学表达式时使用",
            "parameters": {     #参数定义
                "type": "object",
                "properties": {
                    'operation':{   #参数名称
                        'type':'string',    #参数类型
                        'description':'运算符，可选值：+（加）、-（减）、×（乘）、÷（除）'
                        },
                    'a':{
                        'type':'number',
                        'description':'第一个数'                         
                        },
                    'b':{
                        'type':'number',
                        'description':'第二个数',
                        },
                    },
                "required": ['operation', 'a', 'b'],    #["必须提供的参数列表"]
            },
        },
    },
]


class OllamaClient:
    """调用本地ollama模型"""

    def __init__(
        self,
        base_url="http://localhost:11434",
        model="qwen3.5:4b-q4_K_M",
        verbose=True,   # 计时开关
        system=None,    # 设定基础记忆模块
    ):
        self.url = base_url.strip("/")
        self.model = model
        self.verbose = verbose
        self.messages = []      # 多轮对话-记忆库
        if system:      # 判断是否存在基础记忆
            self.messages.append({"role": "system", "content": system})

    def save(self, path="history.json"):
        """把对话历史序列化后写进本地文件，相当于给记忆拍快照"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load(self, path="history.json"):
        """从本地文件反序列化，恢复对话历史到内存"""
        if not os.path.exists(path):  # 确认文件存在
            print("没有历史文件，开始新对话")
            return

        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

        # 防止重复塞 system，避免模型看到两个人设精分
        if not any(m.get("role") == "system" for m in self.messages):
            self.messages.insert(
                0,
                {
                    "role": "system",
                    "content": "你是索蕾娜，安的私人助手，说话自然简洁。",
                },
            )

    def chat(self, message: str) -> str:
        """发送用户消息，拿到回复后存盘"""
        self.messages.append({"role": "user", "content": message})  # 追加我的文本
        url = f"{self.url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.messages,  # 投喂记忆库-多轮对话的核心
            "stream": False,  # 确保非流式输出
            "options": {"num_ctx": 2048},  # 限制上下文窗口
            "tools": TOOLS,     #调用工具
        }

        try:
            start = time.time()  # 开始计时
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()    #抛出4、5类异常给HTTPError
            data = r.json()

            tool_calls = data['message'].get('tool_calls')
            if tool_calls:
                # 支持多个工具
                tool_results = []
                for tool_call in tool_calls:

                    # 拿到模型想调的函数名和参数
                    func_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']

                    # 如果 args 是字符串，json.loads 转字典
                    if isinstance(args, str):
                        args = json.loads(args)


                    # 执行对应的函数
                    if func_name == 'get_current_time':
                        result = get_current_time()
                    elif func_name == "calculator":
                    # calculator 需要 operation, a, b 三个参数
                        result = calculator(
                            operation=args.get('operation'),
                            a=args.get('a'),
                            b=args.get('b'),
                        )
                    else:
                        result = f"不知道这个函数: {func_name}"


                    # 追加所有工具结果
                    tool_results.append({
                        'role':'tool',
                        'content':result,
                        'tool_call_id': tool_call['id']     # 模型返回的每个工具调用都有一个唯一 ID，比如 "call_9xpdxqaf"。Ollama 收到 tool 消息后，会拿这个 ID 去历史里反向查找，自动还原出函数名。即 func_name。
                    })

                # 让模型知道它调了工具
                self.messages.append(data['message'])

                # 把所有工具结果都追加到消息历史
                for tr in tool_results:
                    self.messages.append(tr)

                # 再发一次请求，让模型基于结果说话（这次不带 tools）
                payload2 = {
                    'model':self.model,
                    'messages':self.messages,
                    'stream':False,
                    }

                r2 = requests.post(url, json=payload2, timeout=300)
                r.raise_for_status()
                data2 = r2.json()
                reply2 = data2['message']['content']
                
                # 把最终回复也存入历史并保存
                self.messages.append({"role": "assistant", "content": reply2})
                self.save()

                return reply2
            
            # 确保真正的错误信息被暴露出来
            if "error" in data:
                return f"请求失败：{data['error']}"
            reply = data["message"]["content"]  # 易报错为'message'
            self.messages.append(
                {"role": "assistant", "content": reply}
            )  # 追加模型回答
            cost = time.time() - start  # 结束计时，记录请求用时

            if self.verbose:
                print(f"模型：{self.model}")
                print(f"耗时：{cost:.2f}s")

            self.save()  # 确保在程序崩溃时，减小文本损失

            return reply

        except requests.exceptions.ConnectionError:
            return "请求失败：请确保ollama是否正常运行中"
        except requests.exceptions.Timeout:
            return "请求失败：连接超时，请稍后重试"
        except requests.exceptions.HTTPError as e:
            # 打印Ollama实际返回的内容，方便排查
            print(f"响应内容：{r.text}")
            return f"请求失败：HTTP错误 {e}"
        except (ValueError, KeyError, TypeError) as e:
            # 打印Ollama实际返回的内容，方便排查
            print(f"解析失败，Ollama返回的原始数据：{r.text}")
            return f"请求失败：{e}"

    def clear_history(self):
        """清除记忆库"""
        self.messages = []
        self.save()  # 把空列表写回去，覆盖旧文件


if __name__ == "__main__":
    client = OllamaClient(system="你是索蕾娜，安的私人助手，说话自然简洁。")
    client.load()
    while True:
        prompt = input("你：")
        if prompt == "goodbye":
            print("再见！")
            break
        elif prompt == "clear":
            client.save(path='backup.json')
            client.clear_history()
            print('已清空历史')
        print(client.chat(prompt))

