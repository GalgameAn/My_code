# 索蕾娜 (Sorena) - 个人 AI 秘书（第二阶段）

## 项目简介
这是索蕾娜 AI 项目的第二阶段原型。通过 FastAPI 提供 Web 服务，前端网页可以像真人一样与索蕾娜聊天。
当前版本具备：Ollama 本地模型调用、RAG 长期记忆检索、工具调用、隐私脱敏、异常处理与日志记录。

## 技术栈
- **后端**：Python 3.x, FastAPI, Uvicorn, Pydantic
- **AI 引擎**：Ollama (Qwen 模型), ChromaDB, LangChain (待集成)
- **前端**：原生 HTML/CSS/JS (Fetch API)
- **隐私安全**：正则表达式脱敏 (Privacy.py)

## 目录结构
```text
SoLena_V1/
├── backend/            # FastAPI 后端服务
│   ├── app.py          # 主程序入口
│   └── app.log         # 运行日志（已脱敏）
├── frontend/           # 原生前端页面
│   └── index.html      # 聊天界面
├── OllamaClient_ST.py  # 核心 AI 客户端逻辑
├── Privacy.py          # 隐私脱敏模块
├── Tools.py            # 工具函数（计算器、天气等）
├── Talk.py             # 第一阶段的命令行入口
└── README.md
