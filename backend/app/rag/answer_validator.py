import re


CITATION_RE = re.compile(
    r"\[S(\d+)\]"
)


class AnswerValidator:

    def validate(
        self,
        answer: str,
        source_count: int,
    ) -> bool:

        answer = str(
            answer or ""
        ).strip()

        if not answer:
            return False

        lower = answer.lower()

        # Detect obvious model refusal when evidence exists.
        refusal_patterns = (
            "i don't know",
            "i do not know",
            "cannot answer",
            "can't answer",
            "unable to answer",
        )

        if any(
            pattern in lower
            for pattern in refusal_patterns
        ):
            return False

        citations = [
            int(match)
            for match in CITATION_RE.findall(
                answer
            )
        ]

        if not citations:
            return False

        # Every citation must point to an actual
        # supplied source.
        if any(
            citation < 1
            or citation > source_count
            for citation in citations
        ):
            return False

        return True