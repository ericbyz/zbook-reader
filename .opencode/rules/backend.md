# 后端规则

## API 约定

- 所有 `/api/*` 错误响应统一由 Harness-E 输出 JSON。
- 错误 JSON 字段固定为：`error`、`code`、`status`，需要时可附加 `detail`。
- 健康检查固定为 `/api/health`。
- 内容读取必须带 `project` 和相对 `path`。

## 路径安全

- 内容路径根目录是 `content/`。
- 项目目录必须通过 `HARNESS_E.project_dir(project_id)` 获取。
- 内容文件必须通过 `HARNESS_E.content_file(project_id, path)` 获取。
- 禁止使用未经校验的 `CONTENT_DIR / project / path`。

## 代码运行

- `/api/run` 只执行用户显式点击的 Python 代码块。
- 执行目录固定在 `CONTENT_DIR`。
- 维持超时限制，避免长时间阻塞服务。
- 输出 stdout、stderr、images 三类结果。
