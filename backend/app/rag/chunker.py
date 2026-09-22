from dataclasses import dataclass
import re


@dataclass
class Chunk:
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    text = str(text or "")

    text = text.replace("\x00", " ")

    # Collapse spaces/tabs while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    metadata: dict,
    max_chars: int = 3500,
    overlap: int = 450,
) -> list[Chunk]:

    text = clean_text(text)

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks: list[Chunk] = []
    current = ""

    for paragraph in paragraphs:

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(
                Chunk(
                    text=current,
                    metadata=metadata.copy(),
                )
            )

        tail = (
            current[-overlap:]
            if overlap and current
            else ""
        )

        current = (
            f"{tail}\n\n{paragraph}"
            if tail
            else paragraph
        )

    if current:
        chunks.append(
            Chunk(
                text=current,
                metadata=metadata.copy(),
            )
        )

    return chunks