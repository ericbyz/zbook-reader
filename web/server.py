"""ZBook Web Server - Markdown documentation viewer with code execution."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, NotFound

from web.harness_e import init_harness_e

app = Flask(__name__, static_folder="static", static_url_path="/static")

BASE_DIR = Path(__file__).resolve().parent.parent
CONTENT_DIR = BASE_DIR / "content"
IMPORTED_FILE = CONTENT_DIR / ".imported.json"

# Project display names
PROJECT_META = {
    "ml-tutorial": {"label": "机器学习教程", "icon": "🤖", "desc": "从基础到实战的机器学习完整教程"},
    "ee-tutorial": {"label": "电气工程教程", "icon": "⚡", "desc": "电路分析与电气工程基础"},
    "docs": {"label": "文档", "icon": "📄", "desc": "项目文档与学习笔记"},
}

HARNESS_E = init_harness_e(
    app,
    app_name="zbook-reader",
    base_dir=BASE_DIR,
    content_dir=CONTENT_DIR,
    project_meta=PROJECT_META,
)


def _get_projects():
    """List available content projects."""
    imported = _load_imported()
    projects = []
    if CONTENT_DIR.exists():
        for d in sorted(CONTENT_DIR.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            meta = PROJECT_META.get(d.name, {})
            imp = imported.get(d.name)
            md_files = list(d.rglob("*.md"))
            projects.append({
                "id": d.name,
                "label": meta.get("label") or (imp["label"] if imp else d.name),
                "icon": meta.get("icon", "\U0001f4da"),
                "desc": meta.get("desc", f"{len(md_files)} 篇文档"),
                "count": len(md_files),
                "imported": d.name in imported,
            })
    for book_id, info in imported.items():
        if any(p["id"] == book_id for p in projects):
            continue
        book_path = Path(info["path"]).resolve()
        if not book_path.is_dir():
            continue
        md_files = list(book_path.rglob("*.md"))
        projects.append({
            "id": book_id,
            "label": info.get("label", book_id),
            "icon": "\U0001f4da",
            "desc": info.get("desc", f"{len(md_files)} 篇文档"),
            "count": len(md_files),
            "imported": True,
        })
    return projects


def _scan_files(project_id):
    """Scan markdown files from a specific project."""
    dir_path = _resolve_project_dir(project_id)
    if not dir_path.exists():
        return []
    files = []
    for f in sorted(dir_path.rglob("*.md")):
        rel = f.relative_to(dir_path)
        files.append({
            "path": str(rel),
            "name": f.name,
            "title": _extract_title(f),
        })
    return files


def _extract_title(filepath):
    """Extract first H1 or H2 title from markdown file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
                if line.startswith("## "):
                    return line[3:].strip()
    except Exception:
        pass
    return filepath.name


def _load_imported():
    try:
        return json.loads(IMPORTED_FILE.read_text(encoding="utf-8")) if IMPORTED_FILE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_imported(data):
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_project_dir(project_id):
    imported = _load_imported()
    if project_id in imported:
        p = Path(imported[project_id]["path"]).resolve()
        if not p.is_dir():
            raise NotFound("Imported directory not found")
        return p
    try:
        return HARNESS_E.project_dir(project_id)
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/projects")
def api_projects():
    return jsonify(_get_projects())


@app.route("/api/files/<project_id>")
def api_files(project_id):
    files = _scan_files(project_id)
    if not files and not HARNESS_E.project_dir(project_id).exists():
        raise NotFound("Project not found")
    return jsonify(files)


@app.route("/api/content")
def api_content():
    path = request.args.get("path", "")
    project = request.args.get("project", "")
    if not project:
        raise BadRequest("Project is required")
    try:
        project_dir = _resolve_project_dir(project)
        target = (project_dir / path).resolve()
        target.relative_to(project_dir.resolve())
        full = target
    except ValueError as exc:
        raise BadRequest("Invalid path") from exc
    if not full.exists() or not full.is_file():
        raise NotFound("File not found")
    try:
        text = full.read_text(encoding="utf-8")
        return jsonify({"content": text, "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    """Execute Python code and return output."""
    data = request.get_json(force=True)
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"stdout": "", "stderr": "No code to execute", "images": []})

    with tempfile.TemporaryDirectory() as tmpdir:
        script = os.path.join(tmpdir, "script.py")
        with open(script, "w", encoding="utf-8") as f:
            escaped_tmpdir = tmpdir.replace("\\", "\\\\").replace("'", "\\'")
            f.write(
                "import sys, io, base64, os\n"
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                f"_img_dir = os.path.join(r'{escaped_tmpdir}', 'images')\n"
                "os.makedirs(_img_dir, exist_ok=True)\n"
                "_img_idx = [0]\n"
                "_orig_show = plt.show\n"
                "def _save_and_show(*args, **kwargs):\n"
                "    for fig_num in plt.get_fignums():\n"
                "        fig = plt.figure(fig_num)\n"
                "        path = os.path.join(_img_dir, f'fig_{{_img_idx[0]}}.png')\n"
                "        fig.savefig(path, dpi=100, bbox_inches='tight')\n"
                "        _img_idx[0] += 1\n"
                "    plt.close('all')\n"
                "plt.show = _save_and_show\n\n"
            )
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(CONTENT_DIR),
            )
            stdout = result.stdout
            stderr = result.stderr

            images = []
            img_dir = os.path.join(tmpdir, "images")
            if os.path.exists(img_dir):
                for fname in sorted(os.listdir(img_dir)):
                    fpath = os.path.join(img_dir, fname)
                    with open(fpath, "rb") as imgf:
                        import base64

                        b64 = base64.b64encode(imgf.read()).decode("utf-8")
                        images.append(f"data:image/png;base64,{b64}")

            return jsonify({"stdout": stdout, "stderr": stderr, "images": images})
        except subprocess.TimeoutExpired:
            return jsonify(
                {"stdout": "", "stderr": "Execution timed out (30s)", "images": []}
            )
        except Exception as e:
            return jsonify({"stdout": "", "stderr": str(e), "images": []})


@app.route("/images/<path:filepath>")
def serve_image(filepath):
    """Serve images from content directories."""
    if CONTENT_DIR.exists():
        roots = [d for d in CONTENT_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        full = HARNESS_E.first_existing(roots, filepath)
        if full:
            return send_from_directory(str(full.parent), full.name)
    for info in _load_imported().values():
        book_dir = Path(info["path"]).resolve()
        if book_dir.is_dir():
            try:
                target = (book_dir / filepath).resolve()
                target.relative_to(book_dir)
                if target.exists():
                    return send_from_directory(str(target.parent), target.name)
            except ValueError:
                continue
    raise NotFound("Image not found")


@app.route("/api/import", methods=["POST"])
def api_import():
    data = request.get_json(force=True)
    folder = data.get("path", "").strip()
    if not folder:
        raise BadRequest("请输入文件夹路径")
    src = Path(folder).resolve()
    if not src.is_dir():
        raise BadRequest(f"不是有效的目录: {folder}")
    md_files = list(src.rglob("*.md"))
    if not md_files:
        raise BadRequest("该目录下没有找到 Markdown 文件")
    safe_name = "".join(c if c.isalnum() or c in "-_一-鿿" else "-" for c in src.name).strip("-")
    if not safe_name:
        safe_name = "imported-book"
    imported = _load_imported()
    base = safe_name
    counter = 1
    while safe_name in imported or (CONTENT_DIR / safe_name).exists():
        safe_name = f"{base}-{counter}"
        counter += 1
    imported[safe_name] = {
        "path": str(src),
        "label": data.get("name") or src.name,
        "imported_at": __import__("datetime").datetime.now().isoformat(),
    }
    _save_imported(imported)
    return jsonify({"id": safe_name, "label": imported[safe_name]["label"], "count": len(md_files)})


@app.route("/api/import/<project_id>", methods=["DELETE"])
def api_remove_import(project_id):
    imported = _load_imported()
    if project_id not in imported:
        raise NotFound("Imported project not found")
    del imported[project_id]
    _save_imported(imported)
    return jsonify({"ok": True})


@app.route("/api/browse")
def api_browse():
    path = request.args.get("path", "").strip()
    target = Path(path).resolve() if path else Path.home()
    if not target.is_dir():
        raise BadRequest("Not a directory")
    items = []
    try:
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            is_dir = item.is_dir()
            items.append({"name": item.name, "path": str(item), "is_dir": is_dir})
    except PermissionError:
        pass
    parent = str(target.parent) if str(target) != str(target.parent) else None
    return jsonify({"path": str(target), "parent": parent, "items": items})


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Starting ZBook server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
