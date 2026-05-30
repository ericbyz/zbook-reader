---
name: fix-bug
description: "ZBook 的系统性 Bug 修复流程：先复现，再定位根因，最后最小修复。"
---

# 修复 Bug

## 铁律

```text
没有复现，不改代码。
没有根因，不做修复。
```

## 流程

```text
复现问题 → 提炼最小请求/页面操作 → 定位根因 → 加回归验证 → 最小修复 → 验证通过
```

## Step 1: 复现

优先用最小请求复现：

```bash
curl -s -w '\nHTTP:%{http_code}\n' http://127.0.0.1:9056/api/health
curl -s -w '\nHTTP:%{http_code}\n' 'http://127.0.0.1:9056/api/content?project=ml-tutorial&path=README.md'
```

前端问题先确认：

- 哪个项目卡片？
- 哪篇文档？
- 控制台是否有 JS 错误？
- API 是否返回 JSON 错误？

## Step 2: 定位根因

常见根因：

- 内容目录结构与后端 `CONTENT_DIR` 不一致。
- 相对路径跳转没有带 project。
- 后端路径未做安全校验。
- 前端 fetch 失败后继续读取空数据。
- Python 示例缺少本地依赖。

## Step 3: 最小修复

- 后端错误统一放到 `web/harness_e.py` 或调用点附近处理。
- 前端请求错误统一通过 `apiFetch()` 和 `showHarnessError()`。
- 不顺手重构不相关 UI。

## Step 4: 验证

至少跑：

```bash
python3 -c "import ast, pathlib; [compile(pathlib.Path(p).read_text(), p, 'exec') for p in ['web/server.py','web/asgi.py','web/harness_e.py']]; print('syntax ok')"
node --check /private/tmp/zbook-index.js
```

再跑 API smoke test，见 `.opencode/rules/testing.md`。
