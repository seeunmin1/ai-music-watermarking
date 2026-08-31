#!/usr/bin/env python3
"""Small local provenance detection API.

POST /detect
  - application/json: {"path": "path/to/audio.wav"}
  - multipart/form-data: field name "file"
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from audiomark import op_detect


class DetectionHandler(BaseHTTPRequestHandler):
    server_version = "AudiomarkDetection/0.1"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/detect":
            self._send_json(404, {"error": "not_found"})
            return
        path = None
        try:
            path = self._request_audio_path()
            result = op_detect(path)
            self._send_json(200, {
                "detection": result["detection"].__dict__,
                "metadata": {k: v for k, v in result["metadata"].items() if k != "manifest"},
                "provenance": result["provenance"],
                "ai_detected": result["ai_detected"],
            })
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})
        finally:
            if path and Path(tempfile.gettempdir()).resolve() in path.resolve().parents:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _request_audio_path(self) -> Path:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        if content_type.startswith("application/json"):
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            return Path(data["path"]).resolve()

        body = self.rfile.read(length)
        file_bytes, filename = _extract_multipart_file(body, content_type)
        suffix = Path(filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with tmp:
            tmp.write(file_bytes)
        return Path(tmp.name)


def _extract_multipart_file(body: bytes, content_type: str) -> tuple[bytes, str]:
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("multipart boundary missing")
    boundary = ("--" + content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')).encode()
    for part in body.split(boundary):
        if b"Content-Disposition:" not in part or b'name="file"' not in part:
            continue
        header, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        payload = payload.rsplit(b"\r\n", 1)[0]
        filename = "upload.wav"
        for piece in header.decode("utf-8", errors="ignore").split(";"):
            piece = piece.strip()
            if piece.startswith("filename="):
                filename = piece.split("=", 1)[1].strip().strip('"')
        return payload, filename
    raise ValueError("multipart field 'file' missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Audiomark detection API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DetectionHandler)
    print(f"Detection API listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
