"""Word document text extraction.

.docx — stdlib zipfile + xml, zero external dependencies.
.doc  — antiword system command (like pdftotext for PDF).
"""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile


_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def extract_docx(source_path: str) -> str:
    try:
        zf = zipfile.ZipFile(source_path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid .docx file (bad zip): {source_path!r}") from exc

    with zf:
        try:
            xml_bytes = zf.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"Not a valid .docx file (missing word/document.xml): {source_path!r}") from exc

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for para in root.iter(f"{{{_WORD_NS}}}p"):
        texts = [t.text or "" for t in para.iter(f"{{{_WORD_NS}}}t")]
        line = "".join(texts)
        paragraphs.append(line)

    content = "\n".join(paragraphs)
    if not content.strip():
        raise ValueError(f"No text extracted from .docx: {source_path!r}")
    return content


def extract_doc(source_path: str) -> str:
    antiword = shutil.which("antiword")
    if not antiword:
        raise ValueError(
            ".doc ingestion from source_path requires the antiword command. "
            "Install via: brew install antiword  or  apt install antiword"
        )
    try:
        proc = subprocess.run(
            [antiword, source_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise ValueError(f"antiword failed for source_path {source_path!r}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"antiword timed out for source_path {source_path!r}") from exc

    content = proc.stdout
    if not content.strip():
        raise ValueError(f"antiword extracted no text from source_path {source_path!r}")
    return content
