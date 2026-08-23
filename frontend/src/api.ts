export type RetrievalMode = "bm25" | "dense" | "hybrid";

export type TraceHit = {
  candidate_id: string;
  page_id: string;
  document_id: string;
  page_number: number | null;
  score: number;
  rank: number;
  source: string;
  text: string;
};

export type ContractCheck = {
  name: string;
  status: "pass" | "warning" | "fail" | "not_applicable";
  detail: string;
};

export type RetrievalTrace = {
  schema_version: string;
  trace_id: string;
  question: string;
  mode: RetrievalMode;
  top_k: number;
  dataset_id: string;
  dataset_fingerprint: string;
  text_index_fingerprint: string;
  dense_index_fingerprint: string;
  projection_fingerprint: string;
  benchmark_query_id: string | null;
  stages: {
    sparse_chunks: TraceHit[];
    dense_chunks: TraceHit[];
    sparse_pages: TraceHit[];
    dense_pages: TraceHit[];
    final_pages: TraceHit[];
  };
  context: {
    token_budget: number;
    used_tokens: number;
    truncated: boolean;
    items: Array<{
      candidate_id: string;
      page_id: string;
      document_id: string;
      rank: number;
      source: string;
      text: string;
      token_count: number;
      citation: string;
    }>;
    rendered_context: string;
  };
  copyable_prompt: string;
  timings: {
    sparse_ms: number;
    dense_ms: number;
    projection_and_fusion_ms: number;
    context_ms: number;
    total_ms: number;
  };
  contract_checks: ContractCheck[];
};

export type ValidationResult = {
  schema_version: string;
  trace_id: string;
  valid: boolean;
  checks: ContractCheck[];
  grounding: {
    valid_context_precision: number;
    gold_page_precision: number;
    gold_page_recall: number;
    quote_support_precision: number;
  };
  benchmark: {
    status: "applicable" | "not_applicable";
    reason: string | null;
    query_id: string | null;
    reference_answers: string[];
    answer_quality: { exact_match: number; token_f1: number } | null;
    citation_quality: ValidationResult["grounding"] | null;
    localization_quality: {
      status: string;
      mean_best_iou: number | null;
      predicted_box_count: number;
      gold_box_count: number;
      reason: string | null;
    } | null;
  };
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload = (await response.json()) as { detail?: string } & T;
  if (!response.ok) {
    throw new Error(payload.detail || `API request failed with status ${response.status}`);
  }
  return payload;
}

export function retrieve(payload: {
  question: string;
  mode: RetrievalMode;
  top_k: number;
  context_token_budget: number;
}): Promise<RetrievalTrace> {
  return request<RetrievalTrace>("/api/v1/retrieve", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function validateAnswer(traceId: string, answer: unknown): Promise<ValidationResult> {
  return request<ValidationResult>("/api/v1/validate", {
    method: "POST",
    body: JSON.stringify({ trace_id: traceId, answer }),
  });
}

export function health(): Promise<{ status: string; dense_encoder_status: string }> {
  return request("/api/v1/health");
}
