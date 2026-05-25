"""ZBook Web Server - Markdown documentation viewer with code execution."""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

BASE_DIR = Path(__file__).resolve().parent.parent
CONTENT_DIRS = ["ml-tutorial", "ee-tutorial", "docs"]


def _scan_files():
    """Scan all markdown files from content directories."""
    files = []
    for d in CONTENT_DIRS:
        dir_path = BASE_DIR / d
        if not dir_path.exists():
            continue
        for f in sorted(dir_path.rglob("*.md")):
            rel = f.relative_to(BASE_DIR)
            files.append(
                {
                    "path": str(rel),
                    "name": f.name,
                    "dir": d,
                    "title": _extract_title(f),
                }
            )
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


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/files")
def api_files():
    return jsonify(_scan_files())


@app.route("/api/content")
def api_content():
    path = request.args.get("path", "")
    full = BASE_DIR / path
    if not full.exists() or not str(full.resolve()).startswith(
        str(BASE_DIR.resolve())
    ):
        return jsonify({"error": "File not found"}), 404
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
            # Redirect matplotlib to save figures as base64
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
                cwd=str(BASE_DIR),
            )
            stdout = result.stdout
            stderr = result.stderr

            # Collect generated images
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
    for d in CONTENT_DIRS:
        full = BASE_DIR / d / filepath
        if full.exists():
            return send_from_directory(str(BASE_DIR / d), filepath)
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Starting ZBook server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
