"""Runtime Harness-E integration for ZBook.

This is the runtime half of the harness-e pattern used in this repo. The
documentation and workflow half lives in AGENTS.md and .opencode/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException


@dataclass(frozen=True)
class HarnessEConfig:
    app_name: str
    base_dir: Path
    content_dir: Path
    project_meta: Dict[str, Dict[str, str]]


class HarnessE:
    """Lightweight project harness for error handling and health checks."""

    def __init__(self, config: HarnessEConfig):
        self.config = config

    def register(self, app: Flask) -> None:
        app.config["HARNESS_E"] = self

        @app.get("/api/health")
        def api_health():
            ok, checks = self.validate()
            status = 200 if ok else 503
            return jsonify({
                "ok": ok,
                "app": self.config.app_name,
                "checks": checks,
            }), status

        @app.errorhandler(HTTPException)
        def handle_http_error(error: HTTPException):
            if self._wants_json():
                return self.error_response(
                    error.description or error.name,
                    status_code=error.code or 500,
                    code=error.name.replace(" ", "_").upper(),
                )
            return error

        @app.errorhandler(Exception)
        def handle_unexpected_error(error: Exception):
            app.logger.exception("Unhandled Harness-E error")
            if self._wants_json():
                return self.error_response(
                    "Internal server error",
                    status_code=500,
                    code="INTERNAL_SERVER_ERROR",
                    detail=str(error),
                )
            return "Internal server error", 500

    def validate(self) -> Tuple[bool, Dict[str, Any]]:
        content_dir = self.config.content_dir
        projects = []
        if content_dir.exists():
            projects = [
                p.name
                for p in sorted(content_dir.iterdir())
                if p.is_dir() and not p.name.startswith(".")
            ]
        checks: Dict[str, Any] = {
            "base_dir_exists": self.config.base_dir.exists(),
            "content_dir_exists": content_dir.exists(),
            "content_dir_is_dir": content_dir.is_dir(),
            "projects": projects,
            "project_count": len(projects),
        }
        ok = (
            checks["base_dir_exists"]
            and checks["content_dir_exists"]
            and checks["content_dir_is_dir"]
            and checks["project_count"] > 0
        )
        return bool(ok), checks

    def project_dir(self, project_id: str) -> Path:
        if not project_id or project_id.startswith("."):
            raise ValueError("Invalid project id")
        return self.safe_child(self.config.content_dir, project_id)

    def content_file(self, project_id: str, rel_path: str) -> Path:
        return self.safe_child(self.project_dir(project_id), rel_path)

    def safe_child(self, root: Path, child: str) -> Path:
        root_resolved = root.resolve()
        target = (root_resolved / child).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("Path escapes allowed root") from exc
        return target

    def first_existing(self, roots: Iterable[Path], rel_path: str) -> Path | None:
        for root in roots:
            try:
                candidate = self.safe_child(root, rel_path)
            except ValueError:
                continue
            if candidate.exists():
                return candidate
        return None

    def error_response(
        self,
        message: str,
        *,
        status_code: int,
        code: str = "HARNESS_E_ERROR",
        detail: str | None = None,
    ):
        payload: Dict[str, Any] = {
            "error": message,
            "code": code,
            "status": status_code,
        }
        if detail:
            payload["detail"] = detail
        return jsonify(payload), status_code

    def _wants_json(self) -> bool:
        return request.path.startswith("/api/")


def init_harness_e(
    app: Flask,
    *,
    app_name: str,
    base_dir: Path,
    content_dir: Path,
    project_meta: Dict[str, Dict[str, str]],
) -> HarnessE:
    harness = HarnessE(
        HarnessEConfig(
            app_name=app_name,
            base_dir=base_dir,
            content_dir=content_dir,
            project_meta=project_meta,
        )
    )
    harness.register(app)
    return harness
