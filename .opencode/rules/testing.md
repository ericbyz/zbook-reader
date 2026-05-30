# 测试与验证规则

## 每次后端改动后

```bash
python3 -c "import ast, pathlib; [compile(pathlib.Path(p).read_text(), p, 'exec') for p in ['web/server.py','web/asgi.py','web/harness_e.py']]; print('syntax ok')"
```

## 每次 API 改动后

```bash
python3 - <<'PY'
from web.server import app
client = app.test_client()
checks = [
    ('/api/health', 200),
    ('/api/projects', 200),
    ('/api/files/ml-tutorial', 200),
    ('/api/content?project=ml-tutorial&path=README.md', 200),
    ('/api/content?project=ml-tutorial&path=../README.md', 400),
]
for path, expected in checks:
    res = client.get(path)
    assert res.status_code == expected, (path, res.status_code, expected, res.get_data(as_text=True))
print('api smoke ok')
PY
```

## 每次前端改动后

从 `web/static/index.html` 抽取 `<script>` 内容到临时 JS 文件，再运行：

```bash
node --check /private/tmp/zbook-index.js
```

## 启动验证

```bash
python3 -m uvicorn web.asgi:app --host 127.0.0.1 --port 9056
curl -s http://127.0.0.1:9056/api/health
```
