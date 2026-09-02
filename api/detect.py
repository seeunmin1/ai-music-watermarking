"""Vercel serverless adapter for Audiomark provenance detection."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPOSITORY_ROOT / "cli"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import audiomark  # noqa: E402


MAX_UPLOAD_BYTES = 4 * 1024 * 1024


def _discard_audit_log(*_args, **_kwargs):
    """Serverless requests must not persist an audit trail to the bundle."""


audiomark.audit_log = _discard_audit_log


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        temp_path = None
        try:
            temp_path = self._request_audio_path()
            result = audiomark.op_detect(temp_path)
            detection = result["detection"]
            self._send_json(200, {
                "detection": {
                    "found": bool(detection.found),
                    "record_id": int(detection.record_id),
                    "z": float(detection.z),
                    "confidence": float(detection.confidence),
                    "repetitions": int(detection.repetitions),
                },
                "metadata": {key: value for key, value in result["metadata"].items() if key != "manifest"},
                "provenance": result["provenance"],
                "ai_detected": result["ai_detected"],
                "audio_decode": result["audio_decode"],
                "classifier": result["classifier"],
            })
        except RequestError as error:
            self._send_json(error.status, {"error": str(error)})
        except Exception:
            self._send_json(400, {"error": "Detection failed for this file."})
        finally:
            if temp_path:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _request_audio_path(self) -> Path:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise RequestError("Expected multipart/form-data with a 'file' field.")

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise RequestError("Invalid Content-Length header.") from error
        if content_length <= 0:
            raise RequestError("Audio upload is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            raise RequestError("Audio upload exceeds the 4 MB request limit.", 413)

        body = self.rfile.read(content_length)
        file_bytes, filename, media_type = _extract_multipart_file(body, content_type)
        suffix = Path(filename).suffix or mimetypes.guess_extension(media_type) or ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as upload:
            upload.write(file_bytes)
            return Path(upload.name)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _extract_multipart_file(body: bytes, content_type: str) -> tuple[bytes, str, str]:
    boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;\s]+))", content_type, re.IGNORECASE)
    if not boundary_match:
        raise RequestError("Multipart boundary is missing.")
    boundary_value = boundary_match.group(1) or boundary_match.group(2)
    boundary = ("--" + boundary_value).encode("utf-8")

    for part in body.split(boundary):
        if b"Content-Disposition:" not in part or b'name="file"' not in part:
            continue
        header, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        payload = payload.rsplit(b"\r\n", 1)[0]
        header_text = header.decode("utf-8", errors="ignore")
        filename_match = re.search(r'filename="([^\"]*)"', header_text)
        media_match = re.search(r"Content-Type:\s*([^\r\n;]+)", header_text, re.IGNORECASE)
        return (
            payload,
            filename_match.group(1) if filename_match else "upload.audio",
            media_match.group(1).strip() if media_match else "",
        )

    raise RequestError("Multipart field 'file' is missing.")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
