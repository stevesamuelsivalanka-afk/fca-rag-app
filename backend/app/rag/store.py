import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


LEGAL_SUFFIXES = {
    "plc",
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}


def normalize_firm_name(value: str) -> set[str]:
    """
    Normalize a legal entity name into comparable tokens.

    Examples:

        Barclays plc
        -> {"barclays"}

        Barclays Bank plc
        -> {"barclays", "bank"}

        Barclays Bank UK plc
        -> {"barclays", "bank", "uk"}

    Returns a set intentionally. Callers that need a
    dictionary/set key should use canonical_firm_name().
    """

    value = str(
        value or ""
    ).lower().strip()

    value = value.replace(
        "’",
        "'",
    )

    value = value.replace(
        "–",
        "-",
    )

    value = value.replace(
        "—",
        "-",
    )

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value,
    )

    tokens = [
        token
        for token in value.split()
        if token
        and token not in LEGAL_SUFFIXES
    ]

    return set(tokens)


def canonical_firm_name(value: str) -> str:
    """
    Convert normalized firm tokens into a deterministic
    hashable representation.

    Example:

        Barclays Bank plc
        -> "barclays bank"

    This prevents 'unhashable type: set' errors.
    """

    tokens = normalize_firm_name(
        value
    )

    return " ".join(
        sorted(tokens)
    )


class VectorStore:

    def __init__(
        self,
        directory: str,
    ):
        self.dir = Path(
            directory
        )

        self.dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.dir
            / "index.faiss"
        )

        self.meta_path = (
            self.dir
            / "metadata.json"
        )

        # -------------------------------------------------
        # LOCAL EMBEDDING MODEL
        # -------------------------------------------------

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # -------------------------------------------------
        # EXISTING FAISS INDEX
        # -------------------------------------------------

        self.index = None

        if self.index_path.exists():

            try:

                self.index = faiss.read_index(
                    str(
                        self.index_path
                    )
                )

            except Exception as exc:

                print(
                    "FAISS index error:",
                    exc,
                )

                self.index = None

        # -------------------------------------------------
        # EXISTING METADATA
        # -------------------------------------------------

        self.metadata = []

        if self.meta_path.exists():

            try:

                data = json.loads(
                    self.meta_path.read_text(
                        encoding="utf-8"
                    )
                )

                if isinstance(
                    data,
                    list,
                ):
                    self.metadata = data
                else:
                    print(
                        "WARNING: metadata.json "
                        "does not contain a list."
                    )

            except Exception as exc:

                print(
                    "Metadata load error:",
                    exc,
                )

        print(
            "VectorStore:",
            len(self.metadata),
            "metadata records",
        )

        if self.index is not None:

            print(
                "FAISS vectors:",
                self.index.ntotal,
            )

            if (
                self.index.ntotal
                != len(self.metadata)
            ):

                print(
                    "WARNING: FAISS vector count "
                    "does not match metadata count."
                )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (
                    0,
                    384,
                ),
                dtype="float32",
            )

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        vectors = np.asarray(
            vectors,
            dtype="float32",
        )

        return vectors

    # =====================================================
    # BUILD
    # =====================================================

    def build(
        self,
        chunks,
    ):
        """
        Existing ingestion/build functionality.

        IMPORTANT:
        Normal application startup does not call this.
        The existing FAISS index can therefore remain intact.
        """

        if not chunks:
            return

        vectors = self.embed(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        if vectors.size == 0:
            return

        faiss.normalize_L2(
            vectors
        )

        self.index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        self.index.add(
            vectors
        )

        self.metadata = [
            {
                "text": chunk.text,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        faiss.write_index(
            self.index,
            str(
                self.index_path
            ),
        )

        self.meta_path.write_text(
            json.dumps(
                self.metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # =====================================================
    # FILTER MATCHING
    # =====================================================

    def _matches_filters(
        self,
        metadata: dict,
        filters: dict | None,
        firm_match_mode: str = "exact",
    ) -> bool:

        if not filters:
            return True

        for key, value in filters.items():

            if value is None:
                continue

            # -------------------------------------------------
            # LEGAL ENTITY FILTER
            # -------------------------------------------------

            if key == "firm_normalized":

                actual = normalize_firm_name(
                    metadata.get(
                        "firm_normalized"
                    )
                    or metadata.get(
                        "firm"
                    )
                    or ""
                )

                requested = normalize_firm_name(
                    value
                )

                if not requested:
                    return False

                if (
                    firm_match_mode
                    == "exact"
                ):

                    # IMPORTANT:
                    #
                    # Barclays
                    # != Barclays Bank
                    # != Barclays Bank UK
                    #
                    # This prevents accidental mixing
                    # of separate FCA legal entities.

                    if actual != requested:
                        return False

                else:

                    # Legacy/partial matching mode.
                    if not requested.issubset(
                        actual
                    ):
                        return False

            # -------------------------------------------------
            # SINGLE YEAR
            # -------------------------------------------------

            elif key == "year":

                try:

                    actual_year = int(
                        metadata.get(
                            "year"
                        )
                    )

                    requested_year = int(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    return False

                if (
                    actual_year
                    != requested_year
                ):
                    return False

            # -------------------------------------------------
            # MULTIPLE YEARS
            # -------------------------------------------------

            elif key == "years":

                try:

                    actual_year = int(
                        metadata.get(
                            "year"
                        )
                    )

                    requested_years = {
                        int(year)
                        for year in value
                    }

                except (
                    TypeError,
                    ValueError,
                ):

                    return False

                if (
                    actual_year
                    not in requested_years
                ):
                    return False

            # -------------------------------------------------
            # GENERIC FILTER
            # -------------------------------------------------

            else:

                actual_value = metadata.get(
                    key
                )

                if (
                    str(actual_value)
                    != str(value)
                ):
                    return False

        return True

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        filters: dict | None = None,
        k: int = 8,
        firm_match_mode: str = "exact",
    ):
        """
        Perform exact filtered vector similarity search.

        The dataset is small enough that reconstructing the
        filtered vectors and calculating exact cosine/IP
        similarity is practical.

        No external API is used.
        """

        if (
            self.index is None
            or not self.metadata
        ):
            return []

        if k <= 0:
            return []

        query = str(
            query or ""
        ).strip()

        if not query:
            return []

        # -------------------------------------------------
        # VALIDATE INDEX / METADATA ALIGNMENT
        # -------------------------------------------------

        if (
            self.index.ntotal
            != len(self.metadata)
        ):

            print(
                "WARNING: search aborted because "
                "FAISS and metadata counts differ."
            )

            return []

        # -------------------------------------------------
        # EMBED QUERY
        # -------------------------------------------------

        query_vector = self.embed(
            [query]
        )

        if (
            query_vector.size == 0
        ):
            return []

        faiss.normalize_L2(
            query_vector
        )

        # -------------------------------------------------
        # FILTER CANDIDATES
        # -------------------------------------------------

        candidate_ids = []

        for index, metadata in enumerate(
            self.metadata
        ):

            if self._matches_filters(
                metadata,
                filters,
                firm_match_mode=firm_match_mode,
            ):

                candidate_ids.append(
                    index
                )

        if not candidate_ids:
            return []

        # -------------------------------------------------
        # EXACT FILTERED SIMILARITY
        # -------------------------------------------------

        candidate_vectors = np.vstack(
            [
                self.index.reconstruct(
                    index
                )
                for index in candidate_ids
            ]
        ).astype(
            "float32"
        )

        scores = np.dot(
            candidate_vectors,
            query_vector[0],
        )

        ranked = sorted(
            zip(
                scores,
                candidate_ids,
            ),
            key=lambda item: float(
                item[0]
            ),
            reverse=True,
        )

        # -------------------------------------------------
        # RESULT CREATION
        # -------------------------------------------------

        results = []

        for score, index in ranked[:k]:

            metadata = self.metadata[
                index
            ]

            result = {
                **metadata,
                "score": float(
                    score
                ),
                "_metadata_index": index,
            }

            results.append(
                result
            )

        return results