export type BoundingBox = {
  coordinate_space: "pixel";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  annotator?: number;
};

export type Citation = {
  page_id: string;
  quote: string;
  bounding_box: BoundingBox | null;
};

export type Arm = {
  answer: {
    answer: string;
    refused: boolean;
    citations: Citation[];
  };
  answer_quality: { exact_match: number; token_f1: number };
  citation_quality: {
    valid_context_precision: number;
    gold_page_precision: number;
    gold_page_recall: number;
    quote_support_precision: number;
  };
  context_page_ids: string[];
  gold_page_ids: string[];
  cost: { status: string; estimated_usd: number | null };
  failure_class: string;
  failure_reason: string | null;
  localization_quality: {
    status: string;
    mean_best_iou: number | null;
    predicted_box_count: number;
    gold_box_count: number;
    reason: string | null;
  };
  model: string;
  time_to_response_ms: number;
  usage: {
    input_tokens: number;
    cached_input_tokens: number;
    output_tokens: number;
    reasoning_tokens: number;
    total_tokens: number;
  };
};

export type GoldEvidence = {
  pageId: string;
  relevance: number;
  contentTypes: string[];
  boundingBoxes: BoundingBox[];
};

export type DemoQuery = {
  queryId: string;
  nativeQueryId: number;
  question: string;
  referenceAnswers: string[];
  evidenceTypes: string[];
  queryTypes: string[];
  goldEvidence: GoldEvidence[];
  retrieved: Arm;
  oracle: Arm;
};

export type Page = {
  pageId: string;
  pageNumber: number;
  documentTitle: string;
  width: number;
  height: number;
  imageUrl: string;
};

export type RetrievalSummary = {
  label: string;
  source: string;
  query_count: number;
  ndcg_at_10: number;
  recall_at_10: number;
  mrr: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
};

export type DemoData = {
  schemaVersion: string;
  dataset: string;
  runFingerprint: string;
  recordingNotice: string;
  queries: DemoQuery[];
  pages: Record<string, Page>;
  summaries: {
    retrieval: RetrievalSummary[];
    generation: {
      query_count: number;
      total_estimated_cost_usd: number;
      retrieved: {
        mean_answer_token_f1: number;
        mean_gold_page_recall: number;
        mean_valid_context_precision: number;
        estimated_cost_usd: number;
        total_tokens: number;
        ttr_p50_ms: number;
        ttr_p95_ms: number;
      };
      oracle: {
        mean_answer_token_f1: number;
        mean_gold_page_recall: number;
        mean_valid_context_precision: number;
        estimated_cost_usd: number;
        total_tokens: number;
        ttr_p50_ms: number;
        ttr_p95_ms: number;
      };
    };
    evidence: unknown;
    policyComparison: {
      noise_adjusted_answer_token_f1_delta: number;
      retrieved_context_changed_query_count: number;
      query_count: number;
    };
  };
};
