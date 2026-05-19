"""PDF text extraction via pdftotext (poppler)."""
from __future__ import annotations

import shutil
import subprocess


def extract_pdf(source_path: str) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise ValueError("PDF ingestion from source_path requires the pdftotext command")
    try:
        proc = subprocess.run(
            [pdftotext, source_path, "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise ValueError(f"pdftotext failed for source_path {source_path!r}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"pdftotext timed out for source_path {source_path!r}") from exc

    content = proc.stdout
    if not content.strip():
        raise ValueError(f"pdftotext extracted no text from source_path {source_path!r}")
    return content
