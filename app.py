from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from model import create_analyzer


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
analyzer = create_analyzer()


def predict_payload(raw_text: str) -> tuple[dict, int]:
    text = (raw_text or "").strip()
    if not text:
        return {"error": "Empty text"}, 400
    if len(text) > 3000:
        return {"error": "Text is too long. Please keep entries under 3000 characters."}, 400
    return analyzer.analyze(text), 200


def load_static_file(request_path: str) -> tuple[bytes, str, int]:
    safe_path = request_path.lstrip("/") or "index.html"
    if safe_path.startswith("frontend/"):
        safe_path = safe_path[len("frontend/") :]
    target = (FRONTEND_DIR / safe_path).resolve()
    if not str(target).startswith(str(FRONTEND_DIR.resolve())) or not target.exists() or not target.is_file():
        return b"Not found", "text/plain; charset=utf-8", 404
    content = target.read_bytes()
    mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
        mime = f"{mime}; charset=utf-8"
    return content, mime, 200


try:
    from flask import Flask, jsonify, request, send_from_directory
except ModuleNotFoundError:
    Flask = None


if Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

    @app.after_request
    def add_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": getattr(analyzer, "model_name", "offline-lexicon-analyzer"),
                "model_type": getattr(analyzer, "model_type", "heuristic"),
            }
        )

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return ("", 204)
        payload = request.get_json(silent=True) or {}
        response, status = predict_payload(payload.get("text", ""))
        return jsonify(response), status

    @app.route("/<path:path>")
    def static_files(path: str):
        return send_from_directory(FRONTEND_DIR, path)


class OfflineHandler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(b"", 204, "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            body = json.dumps(
                {
                    "status": "ok",
                    "model": getattr(analyzer, "model_name", "offline-lexicon-analyzer"),
                    "model_type": getattr(analyzer, "model_type", "heuristic"),
                }
            ).encode("utf-8")
            self._send(body, 200, "application/json; charset=utf-8")
            return
        requested = "index.html" if parsed.path in {"/", ""} else parsed.path
        body, mime, status = load_static_file(requested)
        self._send(body, status, mime)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self._send(b'{"error":"Not found"}', 404, "application/json; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send(b'{"error":"Invalid JSON"}', 400, "application/json; charset=utf-8")
            return
        response, status = predict_payload(payload.get("text", ""))
        self._send(json.dumps(response).encode("utf-8"), status, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        return


def run_builtin_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), OfflineHandler)
    print(f"Offline server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    if Flask:
        print("Starting Flask server at http://127.0.0.1:8000")
        app.run(host="127.0.0.1", port=8000, debug=False)
    else:
        print("Flask is not installed, using built-in offline server instead.")
        run_builtin_server()
