from dataclasses import dataclass
import re

@dataclass
class Chunk:
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, metadata: dict, max_chars: int = 3500, overlap: int = 450) -> list[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        candidate = f"{current}\n\n{p}" if current else p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(Chunk(current, metadata.copy()))
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{p}" if tail else p
    if current:
        chunks.append(Chunk(current, metadata.copy()))
    return chunks
