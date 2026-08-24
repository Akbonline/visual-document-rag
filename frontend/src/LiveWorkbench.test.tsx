import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RetrievalTrace } from "./api";
import LiveConsole from "./LiveConsole";
import OutputValidator from "./OutputValidator";

const trace: RetrievalTrace = {
  schema_version: "1",
  trace_id: "a".repeat(32),
  question: "What does ERR-42 mean?",
  mode: "bm25",
  top_k: 1,
  dataset_id: "manual",
  dataset_fingerprint: "b".repeat(64),
  text_index_fingerprint: "c".repeat(64),
  dense_index_fingerprint: "d".repeat(64),
  projection_fingerprint: "e".repeat(64),
  benchmark_query_id: null,
  stages: {
    sparse_chunks: [
      {
        candidate_id: "manual:chunk:1",
        page_id: "manual:page:1",
        document_id: "manual",
        page_number: 1,
        score: 1.4,
        rank: 1,
        source: "bm25",
        text: "ERR-42 means the battery temperature is above its safe range.",
      },
    ],
    dense_chunks: [],
    sparse_pages: [],
    dense_pages: [],
    final_pages: [
      {
        candidate_id: "manual:page:1",
        page_id: "manual:page:1",
        document_id: "manual",
        page_number: 1,
        score: 1.4,
        rank: 1,
        source: "bm25:chunk_to_page:best_rank",
        text: "ERR-42 means the battery temperature is above its safe range.",
      },
    ],
  },
  context: {
    token_budget: 120,
    used_tokens: 10,
    truncated: false,
    items: [
      {
        candidate_id: "manual:page:1",
        page_id: "manual:page:1",
        document_id: "manual",
        rank: 1,
        source: "bm25",
        text: "ERR-42 means the battery temperature is above its safe range.",
        token_count: 10,
        citation: "[manual page 1]",
      },
    ],
    rendered_context: "[manual page 1]\nERR-42 means the battery temperature is above its safe range.",
  },
  copyable_prompt: "QUESTION\nWhat does ERR-42 mean?\n\nEVIDENCE\nERR-42 means the battery temperature is above its safe range.",
  timings: {
    sparse_ms: 2,
    dense_ms: 0,
    projection_and_fusion_ms: 0.2,
    context_ms: 0.1,
    total_ms: 2.3,
  },
  contract_checks: [
    { name: "Context budget", status: "pass", detail: "Used 10 of 120 tokens." },
  ],
};

function response(payload: unknown, ok = true) {
  return Promise.resolve({ ok, status: ok ? 200 : 503, json: () => Promise.resolve(payload) });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("live research workbench", () => {
  it("runs retrieval and exposes the real intermediate trace", async () => {
    const fetchMock = vi.fn().mockImplementation((input: string | URL | Request) => {
      const url = String(input);
      return url.endsWith("/api/v1/health")
        ? response({ status: "ready", dense_encoder_status: "lazy" })
        : response(trace);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<LiveConsole onTrace={() => undefined} onOpenValidator={() => undefined} />);
    await screen.findByLabelText("API ready");
    fireEvent.click(screen.getByText("Use a benchmark question"));
    const runButton = screen.getByRole("button", { name: "Run retrieval pipeline" });
    await waitFor(() => expect(runButton.hasAttribute("disabled")).toBe(false));
    const form = runButton.closest("form") as HTMLFormElement;
    expect(form.checkValidity()).toBe(true);
    fireEvent.submit(form);

    expect(await screen.findByText("2.3 ms")).toBeTruthy();
    expect(screen.getAllByText("Context budget")).toHaveLength(2);
    expect(screen.getByText("Copyable grounded prompt")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("fails visibly instead of substituting demo data when the API is offline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => Promise.reject(new Error("connection refused"))),
    );
    render(<LiveConsole onTrace={() => undefined} onOpenValidator={() => undefined} />);
    await screen.findByLabelText("API offline");
    fireEvent.click(screen.getByText("Use a benchmark question"));
    const runButton = screen.getByRole("button", { name: "Run retrieval pipeline" });
    await waitFor(() => expect(runButton.hasAttribute("disabled")).toBe(false));
    fireEvent.submit(runButton.closest("form") as HTMLFormElement);

    expect(await screen.findByText("API unavailable")).toBeTruthy();
    expect(screen.queryByText("Copyable grounded prompt")).toBeNull();
  });

  it("renders grounded validation and the arbitrary-query correctness boundary", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() =>
        response({
          schema_version: "1",
          trace_id: trace.trace_id,
          valid: true,
          checks: [
            { name: "Quote support", status: "pass", detail: "100.0% supported" },
          ],
          grounding: {
            valid_context_precision: 1,
            gold_page_precision: 0,
            gold_page_recall: 0,
            quote_support_precision: 1,
          },
          benchmark: {
            status: "not_applicable",
            reason: "No exact benchmark match.",
            query_id: null,
            reference_answers: [],
            answer_quality: null,
            citation_quality: null,
            localization_quality: null,
          },
        }),
      ),
    );
    render(<OutputValidator trace={trace} />);
    fireEvent.change(screen.getByLabelText("GeneratedAnswer JSON"), {
      target: {
        value: JSON.stringify({
          answer: "The battery is too hot.",
          refused: false,
          citations: [
            {
              page_id: "manual:page:1",
              quote: "battery temperature is above its safe range",
              bounding_box: null,
            },
          ],
        }),
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Validate grounded output" }));

    await waitFor(() => expect(screen.getByText("Grounded contract passed")).toBeTruthy());
    expect(screen.getByText("Correctness not scored")).toBeTruthy();
  });

  it("renders refusal-only grounding checks as not applicable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() =>
        response({
          schema_version: "1",
          trace_id: trace.trace_id,
          valid: true,
          checks: [
            { name: "Structured output", status: "pass", detail: "Schema valid." },
            { name: "Citation presence", status: "not_applicable", detail: "Refusal." },
            { name: "Citation page membership", status: "not_applicable", detail: "Refusal." },
            { name: "Quote support", status: "not_applicable", detail: "Refusal." },
          ],
          grounding: {
            valid_context_precision: 0,
            gold_page_precision: 0,
            gold_page_recall: 0,
            quote_support_precision: 0,
          },
          benchmark: {
            status: "not_applicable",
            reason: "No exact benchmark match.",
            query_id: null,
            reference_answers: [],
            answer_quality: null,
            citation_quality: null,
            localization_quality: null,
          },
        }),
      ),
    );
    render(<OutputValidator trace={trace} />);
    fireEvent.change(screen.getByLabelText("GeneratedAnswer JSON"), {
      target: {
        value: JSON.stringify({ answer: "Evidence is insufficient.", refused: true, citations: [] }),
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Validate grounded output" }));

    await waitFor(() => expect(screen.getByText("Grounded contract passed")).toBeTruthy());
    expect(screen.getAllByText("N/A").length).toBeGreaterThanOrEqual(4);
  });
});
