from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from vidore_rag.api.models import RetrievalTraceResponse
from vidore_rag.context import EvidenceBundle


@dataclass(frozen=True)
class StoredTrace:
    response: RetrievalTraceResponse
    evidence: EvidenceBundle
    created_at: float


class TraceStore:
    def __init__(self, *, max_entries: int = 200, ttl_seconds: float = 3600) -> None:
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("trace store bounds must be positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, StoredTrace] = OrderedDict()
        self._lock = Lock()

    def put(self, trace: StoredTrace) -> None:
        with self._lock:
            self._prune()
            self._entries[trace.response.trace_id] = trace
            self._entries.move_to_end(trace.response.trace_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get(self, trace_id: str) -> StoredTrace | None:
        with self._lock:
            self._prune()
            trace = self._entries.get(trace_id)
            if trace is not None:
                self._entries.move_to_end(trace_id)
            return trace

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [key for key, value in self._entries.items() if value.created_at < cutoff]
        for key in expired:
            del self._entries[key]
