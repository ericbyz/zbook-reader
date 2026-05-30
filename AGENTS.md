# ZBook Reader - 项目入口

> Markdown 学习资料阅读器：Flask API + 静态前端。内容放在 `content/`，通过本地 Web UI 浏览并运行 Python 示例。
> 核心原则：**先有 evidence，再做修复；项目入口保持精简，深层信息通过路由表按需读取。**

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 后端 | Flask | 提供项目、文件、内容、代码运行 API |
| ASGI 启动 | Uvicorn + WSGI adapter | `web.asgi:app` |
| 前端 | 原生 HTML/CSS/JS | Markdown 渲染、KaTeX、highlight.js、代码运行按钮 |
| 内容 | Markdown | `content/docs`、`content/ee-tutorial`、`content/ml-tutorial` |
| 错误框架 | Harness-E pattern | AGENTS 路由 + evidence 流程 + 运行时错误 harness |

## 目录结构

```text
zbook/
├── AGENTS.md                  # 项目入口 & harness-e 路由
├── README.md                  # 面向用户的简短说明
├── content/                   # Markdown 内容项目
│   ├── docs/
│   ├── ee-tutorial/
│   └── ml-tutorial/
├── web/
│   ├── server.py              # Flask API
│   ├── asgi.py                # Uvicorn 启动适配
│   ├── harness_e.py           # 运行时错误/路径安全 harness
│   └── static/index.html      # 阅读器前端
└── .opencode/                 # Harness-E 工作流与规则
```

## 底线约束

### Always Do

1. 先复现/验证问题，再改代码。
2. 后端路径必须通过 `HarnessE.safe_child()` 或等价方式校验。
3. API 错误必须返回结构化 JSON：`error`、`code`、`status`。
4. 前端请求必须经过 `apiFetch()`，不能静默吞错。
5. 改完后至少跑语法检查、API smoke test、前端 JS 语法检查。

### Ask First

1. 引入 npm/pip 依赖。
2. 改变公开 API 路径或响应字段。
3. 大规模移动内容目录。
4. 删除教程内容或本地编辑器配置。

### Never Do

1. 不要用字符串拼接绕过路径安全。
2. 不要在 API 中返回原始 traceback 给前端。
3. 不要通过删除失败用例来“修复”问题。
4. 不要把 `.obsidian/`、`.playwright-mcp/`、`gitrepo-*` 提交进仓库。

## 路由表：按需查找深层信息

| 你要做的事 | 去哪里看 |
|-----------|---------|
| 修复 Bug | `.opencode/skills/fix-bug/SKILL.md` |
| 检查项目健康度 | `.opencode/commands/health-check.md` |
| 后端/API 约束 | `.opencode/rules/backend.md` |
| 前端约束 | `.opencode/rules/frontend.md` |
| 测试与验证 | `.opencode/rules/testing.md` |
| 用户说明 | `README.md` |
| 后端入口 | `web/server.py` |
| ASGI 启动 | `web/asgi.py` |
| 运行时 Harness-E | `web/harness_e.py` |
| 前端入口 | `web/static/index.html` |

## 核心方法论速查

### Evidence 驱动修复

```text
Bug → 复现命令/请求 → 最小用例 → 根因 → 最小修复 → 验证通过
```

当前项目推荐的最小验证：

```bash
python3 -c "import ast, pathlib; [compile(pathlib.Path(p).read_text(), p, 'exec') for p in ['web/server.py','web/asgi.py','web/harness_e.py']]; print('syntax ok')"
python3 - <<'PY'
from web.server import app
client = app.test_client()
for path, expected in [
    ('/api/health', 200),
    ('/api/projects', 200),
    ('/api/files/ml-tutorial', 200),
    ('/api/content?project=ml-tutorial&path=README.md', 200),
    ('/api/content?project=ml-tutorial&path=../README.md', 400),
]:
    res = client.get(path)
    print(path, res.status_code, 'ok' if res.status_code == expected else 'FAIL')
PY
```

### 启动命令

```bash
python3 -m uvicorn web.asgi:app --host 127.0.0.1 --port 9056
```

## 已知环境注意事项

- 直接运行 `python3 web/server.py 9056` 在当前沙箱里可能因监听端口被拒绝；优先用 Uvicorn。
- `py_compile` 可能尝试写入 `~/Library/Caches` 并失败；本项目用 `compile(..., 'exec')` 做无缓存语法检查。
- 当前 `.git` 是指针文件，指向本地 `gitrepo-final/`。
