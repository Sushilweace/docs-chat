"""Test utilities: PDF builder, SSE parser and a deterministic fake embedder."""

import hashlib
import json
import math
import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
    "what", "which", "who", "how", "do", "does", "it", "its", "for", "with", "that", "this",
}  # fmt: skip
_DIM = 256


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid text PDF with one line of text per page."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
            ).encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Hashed bag-of-words embedding: same words => similar vectors, no model needed."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in _STOPWORDS:
                continue
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM
            vec[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def parse_sse(body: str) -> list[tuple[str, object]]:
    """Parse an SSE response body into ``[(event, json_data), ...]``."""
    events: list[tuple[str, object]] = []
    for frame in body.strip().split("\n\n"):
        event, data = "message", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        events.append((event, json.loads(data)))
    return events
