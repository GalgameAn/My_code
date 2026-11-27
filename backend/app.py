import sys
import os
# 把项目(OllamaClient_ST)根目录(上一级目录)，加入 Python 搜索路径，如此 import 到项目的源码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import logging
# 屏蔽 watchfiles 的噪音
logging.getLogger("watchfiles").setLevel(logging.WARNING)


import uvicorn, requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from Privacy import desensitize
from OllamaClient_ST import OllamaClient


# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backend/app.log", encoding="utf-8"),   # 保存到文件
        logging.StreamHandler()                                     # 打印在终端
    ]
)


# 实例化 FastAPI
app = FastAPI(title="索蕾娜 API", version="0.1.0")


# 实例化大脑
ai_client = OllamaClient()
ai_client.load()


# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers={"*"},
    )


# 定义路由
@app.get("/health")
def health_check():
    """浏览器访问 /health 时，就会触发该函数"""
    # 返回字典，FastAPI 自动转换为 JSON 格式，发给浏览器
    return{"status":"ok", "service":"Solena Backend", "version":"0.1.0"}


# 定义前端接受的数据规则(前端)
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="文本不能为空")   # Field() >> 拦截空文本：最小长度为 1 ，去掉首尾空格  
    session_id: str = "default"     # 此处等号代表默认值


# 定义 POST 路由(后端)
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """现在的逻辑：收到请求 >> 交给大脑处理 >> 拿到回复 >> 返回给前端"""

    # 记录用户日志
    safe_user_msg = desensitize(request.message)
    logging.info(f"收到请求 [Session:{request.session_id}] 内容：{safe_user_msg}")


    try:
        reply = ai_client.chat(request.message)

        # 记录回复日志
        safe_reply = desensitize(reply)
        logging.info(f"回复完成 [Session:{request.session_id}] 内容：{safe_reply}")

        return{"reply":reply, "status":"success"}
    
    except requests.exceptions.ConnectionError:
        logging.error("Ollama 服务未启动或连接失败")
        raise HTTPException(status_code=503, detail="Ollama 似乎未启动，请检查后台。")
    except requests.exceptions.Timeout:
        logging.error("Ollama 响应超时")
        raise HTTPException(status_code=504, detail="思考超时，请重试。")
    except Exception as e:
        logging.error(f"发生未知异常：{e}")
        raise HTTPException(status_code=500, detail="服务器内部出现未知错误")

    
if __name__ == "__main__":
    # 127.0.0.1 >> 只允许本机访问(后续部署可改为0.0.0.0)；8000 >> 端口号；reload=Ture >> 热更新：修改代码，自动重启
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)