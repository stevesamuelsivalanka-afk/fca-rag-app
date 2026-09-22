import json
import hashlib
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.entity_resolution import normalize_entity_name


def normalize_text(value) -> str:
    return normalize_entity_name(value)


def organization_tokens(value) -> set[str]:
    return set(normalize_text(value).split())


class VectorStore:

    def __init__(self, directory: str):

        self.dir = Path(directory)
        self.dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "metadata.json"

        self._model: SentenceTransformer | None = None
        self.load_issues: list[str] = []
        self.index = None
        self.metadata: list[dict] = []

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as exc:
                self.load_issues.append(f"FAISS index could not be read: {exc}")
        else:
            self.load_issues.append("FAISS index file is missing.")

        if self.meta_path.exists():
            try:
                loaded = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, list):
                    raise ValueError("metadata root must be a JSON list")
                self.metadata = loaded
            except Exception as exc:
                self.load_issues.append(f"Metadata could not be read: {exc}")
        else:
            self.load_issues.append("Metadata file is missing.")

        self.validation = self.validate()

    @property
    def size(self) -> int:

        if self.index is None:
            return 0

        return int(
            self.index.ntotal
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def validate(self) -> dict:
        issues = list(self.load_issues)
        index_count = self.size
        metadata_count = len(self.metadata)

        if self.index is not None and index_count != metadata_count:
            issues.append(
                f"FAISS/metadata count mismatch ({index_count} vectors, "
                f"{metadata_count} records)."
            )

        required = ("firm", "firm_normalized", "year", "url", "page", "text")
        seen_chunks: set[tuple[str, str, int, str]] = set()
        duplicate_count = 0
        available_years: set[int] = set()

        for position, item in enumerate(self.metadata):
            if not isinstance(item, dict):
                issues.append(f"Metadata record {position} is not an object.")
                continue
            for field in required:
                value = item.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(f"Metadata record {position} is missing {field}.")

            try:
                year = int(item.get("year"))
                if not 1900 <= year <= 2100:
                    raise ValueError
                available_years.add(year)
            except (TypeError, ValueError):
                issues.append(f"Metadata record {position} has an invalid year.")

            try:
                page = int(item.get("page"))
                if page < 1:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"Metadata record {position} has an invalid page.")

            amount_text = str(item.get("amount_text") or "")
            amount = item.get("amount")
            if amount_text and amount is not None:
                try:
                    if float(amount) < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    issues.append(f"Metadata record {position} has a malformed amount.")

            text = str(item.get("text") or "").strip()
            if text:
                try:
                    key = (
                        str(item.get("url") or ""),
                        str(item.get("firm_normalized") or ""),
                        int(item.get("page") or 0),
                        hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                    if key in seen_chunks:
                        duplicate_count += 1
                    seen_chunks.add(key)
                except (TypeError, ValueError):
                    pass

        if duplicate_count:
            issues.append(f"Found {duplicate_count} duplicate chunk records.")
        if self.index is None and metadata_count:
            issues.append("Metadata records have no corresponding FAISS index.")
        if self.index is not None and metadata_count > index_count:
            issues.append("Metadata contains records without FAISS vectors.")
        if self.index is not None and index_count > metadata_count:
            issues.append("FAISS contains vectors without metadata records.")

        return {
            "valid": not issues,
            "issues": list(dict.fromkeys(issues)),
            "indexed_chunks": index_count,
            "metadata_count": metadata_count,
            "faiss_count": index_count,
            "available_years": sorted(available_years),
            "duplicate_chunks": duplicate_count,
        }

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return np.asarray(
            vectors,
            dtype="float32",
        )

    # ---------------------------------------------------------
    # Build
    # ---------------------------------------------------------

    def build(self, chunks):

        if not chunks:
            raise ValueError(
                "Cannot build an empty vector store."
            )

        new_metadata = [
            {
                "text": chunk.text,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        old_texts = [str(item.get("text") or "") for item in self.metadata]
        new_texts = [str(item.get("text") or "") for item in new_metadata]
        if self.index is None or old_texts != new_texts or self.size != len(new_texts):
            vectors = self.embed(new_texts)
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            temporary_index = self.index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(temporary_index))
            temporary_index.replace(self.index_path)
            self.index = index

        self.metadata = new_metadata
        temporary_metadata = self.meta_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_metadata.replace(self.meta_path)
        self.load_issues = []
        self.validation = self.validate()

    # ---------------------------------------------------------
    # Metadata filtering
    # ---------------------------------------------------------

    def _matches(
        self,
        metadata: dict,
        filters: dict | None,
    ) -> bool:

        if not filters:
            return True

        for key, value in filters.items():

            if value is None:
                continue

            if key == "firm_entity":
                actual = metadata.get("firm_normalized") or metadata.get("firm")
                if normalize_text(actual) != normalize_text(value):
                    return False

            elif key == "entity_type":
                if value == "bank" and not re.search(
                    r"\bbank\b",
                    str(metadata.get("firm") or ""),
                    re.IGNORECASE,
                ):
                    return False

            # -------------------------------------------------
            # Backward-compatible firm filter
            # -------------------------------------------------

            elif key == "firm_normalized":

                actual = normalize_text(
                    metadata.get(
                        "firm_normalized",
                        metadata.get(
                            "firm",
                            "",
                        ),
                    )
                )

                requested = normalize_text(
                    value
                )

                if actual != requested:
                    return False

            # -------------------------------------------------
            # Single year
            # -------------------------------------------------

            elif key == "year":

                try:

                    if int(
                        metadata.get("year")
                    ) != int(value):

                        return False

                except (
                    TypeError,
                    ValueError,
                ):

                    return False

            # -------------------------------------------------
            # Multiple years
            # -------------------------------------------------

            elif key == "years":

                try:

                    actual_year = int(
                        metadata.get("year")
                    )

                    requested_years = {
                        int(year)
                        for year in value
                    }

                    if actual_year not in requested_years:
                        return False

                except (
                    TypeError,
                    ValueError,
                ):

                    return False

            # -------------------------------------------------
            # Generic equality
            # -------------------------------------------------

            else:

                if str(
                    metadata.get(key)
                ) != str(value):

                    return False

        return True

    # ---------------------------------------------------------
    # Metadata-only query
    # ---------------------------------------------------------

    def filter_metadata(
        self,
        filters: dict | None = None,
    ) -> list[dict]:

        if not self.metadata:
            return []

        return [
            item
            for item in self.metadata
            if self._matches(
                item,
                filters,
            )
        ]

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict | None = None,
        k: int = 8,
    ) -> list[dict]:

        if (
            self.index is None
            or not self.metadata
            or k <= 0
            or not self.validation["valid"]
        ):
            return []

        candidate_ids = [
            idx
            for idx, metadata
            in enumerate(self.metadata)
            if self._matches(
                metadata,
                filters,
            )
        ]

        if not candidate_ids:
            return []

        query_vector = self.embed(
            [query]
        )

        faiss.normalize_L2(
            query_vector
        )

        candidate_vectors = np.vstack(
            [
                self.index.reconstruct(idx)
                for idx in candidate_ids
            ]
        ).astype("float32")

        semantic_scores = np.dot(
            candidate_vectors,
            query_vector[0],
        )

        query_tokens = set(re.findall(r"[a-z0-9]+", normalize_text(query)))
        lexical_scores = np.asarray(
            [
                len(
                    query_tokens.intersection(
                        re.findall(
                            r"[a-z0-9]+",
                            normalize_text(self.metadata[idx].get("text")),
                        )
                    )
                )
                / max(1, len(query_tokens))
                for idx in candidate_ids
            ],
            dtype="float32",
        )
        hybrid_scores = 0.8 * semantic_scores + 0.2 * lexical_scores

        ranked = sorted(
            zip(hybrid_scores, semantic_scores, candidate_ids),
            key=lambda item: item[0],
            reverse=True,
        )

        results = []

        for hybrid_score, score, idx in ranked[:k]:

            metadata = self.metadata[idx]
            source_key = "|".join(
                (
                    str(metadata.get("year") or ""),
                    normalize_text(metadata.get("firm_normalized") or metadata.get("firm")),
                    str(metadata.get("url") or ""),
                    str(metadata.get("page") or ""),
                    str(metadata.get("text") or ""),
                )
            )

            results.append(
                {
                    **metadata,
                    "score": float(score),
                    "hybrid_score": float(hybrid_score),
                    "source_id": hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16],
                }
            )

        return results