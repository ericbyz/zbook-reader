---
name: health-check
description: "检查 ZBook Reader 项目健康度：API、内容目录、前端语法、启动方式。"
---

# 项目健康度检查

## 检查维度

### 1. API 健康度

- [ ] `/api/health` 返回 200 且 `ok: true`
- [ ] `/api/projects` 返回 docs、ee-tutorial、ml-tutorial
- [ ] `/api/files/<project>` 返回文档列表
- [ ] `/api/content` 能读取正常文档
- [ ] 路径逃逸请求返回 400

### 2. 前端健康度

- [ ] `web/static/index.html` 中脚本语法通过 `node --check`
- [ ] 所有 API 请求使用 `apiFetch()`
- [ ] 文档加载失败有可见错误提示

### 3. 后端健康度

- [ ] `web/server.py`、`web/asgi.py`、`web/harness_e.py` 语法通过
- [ ] 路径安全通过 Harness-E
- [ ] 未把 traceback 暴露给普通 API 响应

### 4. 内容健康度

- [ ] `content/docs` 存在
- [ ] `content/ee-tutorial` 存在且包含 README
- [ ] `content/ml-tutorial` 存在且包含 README

## 推荐命令

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

## 输出模板

```markdown
## ZBook 项目健康度报告

**日期**: YYYY-MM-DD
**总体评分**: 绿色 / 黄色 / 红色

### API
- 结果:
- 问题:

### 前端
- 结果:
- 问题:

### 后端
- 结果:
- 问题:

### 内容
- 结果:
- 问题:

### 建议操作
1.
2.
3.
```
