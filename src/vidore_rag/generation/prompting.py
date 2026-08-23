from __future__ import annotations

from vidore_rag.context import EvidenceBundle
from vidore_rag.generation.models import GeneratedAnswer, ProviderRequest

INSTRUCTIONS = """You answer questions only from the supplied document evidence.
If the evidence cannot answer the question, set refused=true and answer briefly that the
evidence is insufficient. Otherwise set refused=false. Cite only exact page_id values supplied
in the evidence and include a short verbatim quote. A refusal may cite evidence that was
inspected for transparency, but do not claim a citation proves that an answer is absent.
Do not invent coordinates;
bounding_box must be null unless the evidence explicitly supplies pixel coordinates."""


def build_request(
    *,
    question: str,
    evidence: EvidenceBundle,
    max_output_tokens: int,
) -> ProviderRequest:
    page_ids = [item.page_id for item in evidence.items]
    prompt = (
        f"Question:\n{question}\n\n"
        f"Allowed page IDs:\n{page_ids}\n\n"
        f"Evidence:\n{evidence.rendered_context}"
    )
    return ProviderRequest(
        instructions=INSTRUCTIONS,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        output_schema=GeneratedAnswer.model_json_schema(),
    )
