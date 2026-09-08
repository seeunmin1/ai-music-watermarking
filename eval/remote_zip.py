"""Read selected members out of a remote ZIP without downloading the archive.

The public AI-music corpora ship as multi-gigabyte ZIP parts, but a validation
sample only needs tens of tracks. ZIP stores its central directory at the end of
the file, so with HTTP range requests we can read the directory, then fetch just
the byte ranges for the members we want.

`HttpRangeFile` presents a seekable file-like object over range requests, which
is enough for the stdlib `zipfile` module to work unmodified.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from typing import Iterable

USER_AGENT = "audiomark-eval/1.0"
# Read ahead in chunks so the central-directory scan is not one request per read.
DEFAULT_CHUNK = 1 << 20


def _ssl_context() -> ssl.SSLContext:
    """Framework Python on macOS ships no CA bundle; use certifi's when present."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT)


def resolve_redirects(url: str, timeout: float = 60.0) -> str:
    """Follow redirects once so range requests hit the final storage URL."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with _urlopen(request, timeout) as response:
        return response.geturl()


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file backed by HTTP range requests."""

    def __init__(self, url: str, timeout: float = 120.0, chunk_size: int = DEFAULT_CHUNK):
        self.url = resolve_redirects(url, timeout)
        self.timeout = timeout
        self.chunk_size = chunk_size
        self._pos = 0
        self._cache: tuple[int, bytes] | None = None  # (start_offset, data)
        self.size = self._content_length()

    def _content_length(self) -> int:
        request = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with _urlopen(request, self.timeout) as response:
            length = response.headers.get("Content-Length")
        if not length:
            raise OSError("server did not report Content-Length; cannot range-read")
        return int(length)

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch the inclusive byte range [start, end]."""
        end = min(end, self.size - 1)
        if start > end:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
        )
        with _urlopen(request, self.timeout) as response:
            if response.status != 206:
                raise OSError(f"server ignored the range request (HTTP {response.status})")
            return response.read()

    # --- io.RawIOBase interface -------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self._pos
        size = min(size, self.size - self._pos)
        if size <= 0:
            return b""

        if self._cache is not None:
            start, data = self._cache
            if start <= self._pos and self._pos + size <= start + len(data):
                offset = self._pos - start
                self._pos += size
                return data[offset : offset + size]

        span = max(size, self.chunk_size)
        data = self._fetch(self._pos, self._pos + span - 1)
        self._cache = (self._pos, data)
        self._pos += size
        return data[:size]

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def open_remote_zip(url: str, timeout: float = 120.0) -> zipfile.ZipFile:
    """Open a remote ZIP for selective member extraction."""
    return zipfile.ZipFile(HttpRangeFile(url, timeout))


def extract_members(
    archive: zipfile.ZipFile,
    names: Iterable[str],
    destination,
    flatten: bool = True,
) -> list:
    """Extract named members, returning the paths written."""
    from pathlib import Path

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        target = destination / (Path(name).name if flatten else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(name) as source:
            target.write_bytes(source.read())
        written.append(target)
    return written
