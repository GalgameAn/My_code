import time
import os
import json
import requests


class OllamaClient:
    """调用本地ollama模型"""

    def __init__(
        self,
        base_url="http://localhost:11434",
        model="qwen3.5:4b-q4_K_M",
        verbose=True,  # 计时开关
        system=None,  # 设定基础记忆模块
        ):
        self.url = base_url.strip("/")
        self.model = model
        self.verbose = verbose
        self.messages = []  # 多轮对话-记忆库
        if system:  # 判断是否存在基础记忆
            self.messages.append({"role": "system", "content": system})


    def save(self, path="history.json"):
        """把对话历史序列化后写进本地文件，相当于给记忆拍快照"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)


    def load(self, path="history.json"):
        """从本地文件反序列化，恢复对话历史到内存"""
        if not os.path.exists(path):  #确认文件存在
            print("没有历史文件，开始新对话")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            self.messages = json.load(f)

        # 防止重复塞 system，避免模型看到两个人设精分
        if not any(m.get('role') == 'system' for m in self.messages):
            self.messages.insert(0, {
                'role':'system',
                'content':'你是索蕾娜，安的私人助手，说话自然简洁。'
            })


    def chat(self, message: str) -> str:
        """发送用户消息，拿到回复后存盘"""
        self.messages.append({"role": "user", "content": message})  # 追加我的文本
        url = f"{self.url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self.messages,  # 投喂记忆库-多轮对话的核心
            "stream": False,  # 确保非流式输出
            "options": {"num_ctx": 2048},  #限制上下文窗口
        }

        try:
            start = time.time()  # 请求开始计时
            r = requests.post(url, json=payload, timeout=180)
            data = r.json()
            #确保真正的错误信息被暴露出来
            if 'error' in data:
                return f"请求失败：{data['error']}"
            reply = data["message"]["content"]  #易报错为'message'
            self.messages.append(
                {"role": "assistant", "content": reply}
            )  # 追加模型回答
            cost = time.time() - start  # 记录请求用时

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
            #打印Ollama实际返回的内容，方便排查
            print(f"解析失败，Ollama返回的原始数据：{r.text}")
            return f"请求失败：{e}"
      

    def clear_history(self):
        """清除记忆库"""
        self.messages = []
        self.save()  # 把空列表写回去，覆盖旧文件


if __name__ == "__main__":
    client = OllamaClient( system="你是索蕾娜，安的私人助手，说话自然简洁。")
    client.load()
    while True:
        prompt = input("你：")
        if prompt == 'goodbye':
            print('再见！')
            break
        print(client.chat(prompt))