"""Dispatch extract_source_text by file extension."""
from __future__ import annotations

import hashlib
import os


def extract_source_text(source_path: str) -> tuple[str, str, int]:
    """Read source_path and return (text, sha256_of_raw, size_of_raw)."""
    if not os.path.isfile(source_path):
        raise ValueError(f"source_path does not exist or is not a file: {source_path!r}")

    with open(source_path, "rb") as f:
        raw = f.read()

    source_hash = hashlib.sha256(raw).hexdigest()
    source_size = len(raw)
    ext = os.path.splitext(source_path)[1].lower()

    if ext == ".pdf":
        from .pdf import extract_pdf
        return extract_pdf(source_path), source_hash, source_size

    if ext == ".docx":
        from .docx import extract_docx
        return extract_docx(source_path), source_hash, source_size

    if ext == ".doc":
        from .docx import extract_doc
        return extract_doc(source_path), source_hash, source_size

    try:
        return raw.decode("utf-8"), source_hash, source_size
    except UnicodeDecodeError as exc:
        raise ValueError(
            "source_path is not valid UTF-8 text. "
            "Supported binary formats: .pdf, .docx, .doc"
        ) from exc
