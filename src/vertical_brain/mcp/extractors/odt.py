"""ODT (OpenDocument Text) extraction via stdlib zipfile + xml."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"


def extract_odt(source_path: str) -> str:
    try:
        zf = zipfile.ZipFile(source_path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid .odt file (bad zip): {source_path!r}") from exc

    with zf:
        try:
            xml_bytes = zf.read("content.xml")
        except KeyError as exc:
            raise ValueError(f"Not a valid .odt file (missing content.xml): {source_path!r}") from exc

    root = ET.fromstring(xml_bytes)
    lines: list[str] = []
    for p in root.iter(f"{{{_TEXT_NS}}}p"):
        parts: list[str] = []
        if p.text:
            parts.append(p.text)
        for child in p:
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        lines.append("".join(parts))

    content = "\n".join(lines)
    if not content.strip():
        raise ValueError(f"No text extracted from .odt: {source_path!r}")
    return content
