from __future__ import annotations

from pydantic import BaseModel, Field

from vidore_rag.document_ir import SearchHit


class EvidenceItem(BaseModel):
    candidate_id: str
    page_id: str
    document_id: str
    rank: int = Field(ge=1)
    source: str
    text: str
    token_count: int = Field(ge=1)
    citation: str


class EvidenceBundle(BaseModel):
    token_budget: int = Field(ge=1)
    used_tokens: int = Field(ge=0)
    truncated: bool
    items: list[EvidenceItem]
    rendered_context: str


class ContextBuilder(BaseModel):
    token_budget: int = Field(default=1200, ge=1)

    def build(self, hits: list[SearchHit]) -> EvidenceBundle:
        items: list[EvidenceItem] = []
        used = 0
        truncated = False
        seen: set[str] = set()
        for hit in hits:
            if hit.candidate_id in seen:
                continue
            seen.add(hit.candidate_id)
            tokens = hit.text.split()
            remaining = self.token_budget - used
            if remaining <= 0:
                truncated = True
                break
            if len(tokens) > remaining:
                if not items:
                    tokens = tokens[:remaining]
                    truncated = True
                else:
                    truncated = True
                    continue
            text = " ".join(tokens)
            citation = f"[{hit.document_id} page {hit.page_id.rsplit(':', 1)[-1]}]"
            items.append(
                EvidenceItem(
                    candidate_id=hit.candidate_id,
                    page_id=hit.page_id,
                    document_id=hit.document_id,
                    rank=hit.rank,
                    source=hit.source,
                    text=text,
                    token_count=len(tokens),
                    citation=citation,
                )
            )
            used += len(tokens)
        rendered = "\n\n".join(f"{item.citation}\n{item.text}" for item in items)
        return EvidenceBundle(
            token_budget=self.token_budget,
            used_tokens=used,
            truncated=truncated,
            items=items,
            rendered_context=rendered,
        )
