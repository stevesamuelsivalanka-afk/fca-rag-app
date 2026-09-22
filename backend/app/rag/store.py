import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "metadata.json"

        # Local embedding model - no OpenAI API required
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = (
            faiss.read_index(str(self.index_path))
            if self.index_path.exists()
            else None
        )

        self.metadata = (
            json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.meta_path.exists()
            else []
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype="float32")

    def build(self, chunks):
        vectors = self.embed([c.text for c in chunks])

        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.metadata = [
            {"text": c.text, **c.metadata}
            for c in chunks
        ]

        faiss.write_index(self.index, str(self.index_path))

        self.meta_path.write_text(
            json.dumps(
                self.metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8",
        )

    def search(
        self,
        query: str,
        filters: dict | None = None,
        k: int = 8
    ):
        if self.index is None or not self.metadata:
            return []

        # Embed and normalize query
        q = self.embed([query])
        faiss.normalize_L2(q)

        # Find candidate document IDs using metadata filters first
        candidate_ids = []

        for idx, m in enumerate(self.metadata):
            if not filters:
                candidate_ids.append(idx)
                continue

            matched = True


            for key, value in filters.items():
                if value is None:
                    continue

                if key == "firm_normalized":
                    actual_firm = str(
                        m.get("firm_normalized") or m.get("firm") or ""
                    ).lower().strip()

                    requested_firm = str(value).lower().strip()

                    if requested_firm not in actual_firm:
                        matched = False
                        break

                elif key == "year":
                    try:
                        actual_year = int(m.get("year"))

                        if actual_year != int(value):
                            matched = False
                            break

                    except (TypeError, ValueError):
                        matched = False
                        break

                elif key == "years":
                    try:
                        actual_year = int(m.get("year"))
                        requested_years = {int(v) for v in value}

                        if actual_year not in requested_years:
                            matched = False
                            break

                    except (TypeError, ValueError):
                        matched = False
                        break

                else:
                    if str(m.get(key)) != str(value):
                        matched = False
                        break

            # for key, value in filters.items():
            #     if value is None:
            #         continue

            #     actual = m.get(key)

            #     if key == "firm_normalized":
            #         actual_firm = str(actual or "").lower().strip()
            #         requested_firm = str(value).lower().strip()

            #         if requested_firm not in actual_firm:
            #             matched = False
            #             break

            #     elif key == "year":
            #         try:
            #             if int(actual) != int(value):
            #                 matched = False
            #                 break
            #         except (TypeError, ValueError):
            #             matched = False
            #             break

            #     elif key == "years":
            #         try:
            #             requested_years = [int(v) for v in value]

            #             if int(actual) not in requested_years:
            #                 matched = False
            #                 break

            #         except (TypeError, ValueError):
            #             matched = False
            #             break

            #     else:
            #         if str(actual) != str(value):
            #             matched = False
            #             break

            # for key, value in filters.items():
            #     if value is None:
            #         continue

            #     actual = m.get(key)

            #     if key == "firm_normalized":
            #         actual_firm = str(actual or "").lower()
            #         requested_firm = str(value).lower()

            #         if (
            #             actual_firm != requested_firm
            #             and requested_firm not in actual_firm
            #         ):
            #             matched = False
            #             break

            #     elif key == "year":
            #         try:
            #             if int(actual) != int(value):
            #                 matched = False
            #                 break
            #         except (TypeError, ValueError):
            #             matched = False
            #             break

            #     elif key == "years":
            #         try:
            #             requested_years = [int(v) for v in value]

            #             if int(actual) not in requested_years:
            #                 matched = False
            #                 break

            #         except (TypeError, ValueError):
            #             matched = False
            #             break

            #     else:
            #         if str(actual) != str(value):
            #             matched = False
            #             break
                                
            #     # elif key == "year":
            #     #     try:
            #     #         if int(actual) != int(value):
            #     #             matched = False
            #     #             break
            #     #     except (TypeError, ValueError):
            #     #         matched = False
            #     #         break

            #     # else:
            #     #     if str(actual) != str(value):
            #     #         matched = False
            #     #         break

            if matched:
                candidate_ids.append(idx)

        # No documents match the requested metadata filters
        if not candidate_ids:
            return []

        # Calculate similarity only for matching candidates
        candidate_vectors = np.vstack([
            self.index.reconstruct(idx)
            for idx in candidate_ids
        ]).astype("float32")

        scores = np.dot(candidate_vectors, q[0])

        # Highest similarity first
        ranked = sorted(
            zip(scores, candidate_ids),
            key=lambda x: x[0],
            reverse=True
        )

        out = []

        for score, idx in ranked[:k]:
            m = self.metadata[idx]

            out.append({
                **m,
                "score": float(score)
            })

        return out
   