# zbook

学习资料与小型工具合集。

## 内容

- `content/ee-tutorial/`: 电子工程教程，从电路基础到数字电路、模拟电路。
- `content/ml-tutorial/`: 机器学习教程，从数学基础到经典算法与实践流程。
- `content/docs/`: 其他学习指南与笔记。
- `web/`: 用于浏览内容的简单 Web 页面与服务脚本。

## 使用

直接阅读各目录下的 Markdown 文件即可。`web/` 目录包含一个本地静态页面和服务脚本，可按需要启动查看。

## Harness-E

项目已按 `harness-e pattern` 接入：`AGENTS.md` 作为精简入口和路由表，`.opencode/` 存放修复 Bug、健康检查、前后端约束等工作流；`web/harness_e.py` 是运行时错误与路径安全层。

- 健康检查：`/api/health`
- ASGI 启动入口：`web.asgi:app`
- 项目入口：`AGENTS.md`
