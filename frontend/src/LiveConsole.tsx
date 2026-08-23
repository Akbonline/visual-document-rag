import { type FormEvent, useEffect, useMemo, useState } from "react";
import { health, retrieve, type RetrievalMode, type RetrievalTrace, type TraceHit } from "./api";

const BENCHMARK_EXAMPLE =
  "What was Romania’s employment rate for people aged 15 to 24 in 2020?";

type StageName = keyof RetrievalTrace["stages"];

const STAGE_LABELS: Record<StageName, string> = {
  sparse_chunks: "BM25 chunks",
  dense_chunks: "Dense chunks",
  sparse_pages: "BM25 pages",
  dense_pages: "Dense pages",
  final_pages: "Final pages",
};

function HitList({ hits }: { hits: TraceHit[] }) {
  if (!hits.length) {
    return <div className="empty-stage">This retrieval arm was not used for the selected mode.</div>;
  }
  return (
    <div className="trace-hits">
      {hits.slice(0, 12).map((hit) => (
        <article key={`${hit.source}-${hit.candidate_id}-${hit.rank}`}>
          <div className="trace-hit-head">
            <span>#{hit.rank}</span>
            <strong>Page {hit.page_number ?? hit.page_id.split(":").at(-1)}</strong>
            <code>{hit.score.toFixed(5)}</code>
          </div>
          <p>{hit.text}</p>
          <small>{hit.source} · {hit.candidate_id}</small>
        </article>
      ))}
    </div>
  );
}

export default function LiveConsole({
  onTrace,
  onOpenValidator,
}: {
  onTrace: (trace: RetrievalTrace) => void;
  onOpenValidator: () => void;
}) {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(10);
  const [budget, setBudget] = useState(1200);
  const [trace, setTrace] = useState<RetrievalTrace | null>(null);
  const [stage, setStage] = useState<StageName>("final_pages");
  const [status, setStatus] = useState<"checking" | "ready" | "offline">("checking");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    health().then(() => setStatus("ready")).catch(() => setStatus("offline"));
  }, []);

  const activeHits = useMemo(() => trace?.stages[stage] ?? [], [trace, stage]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const next = await retrieve({
        question: question.trim(),
        mode,
        top_k: topK,
        context_token_budget: budget,
      });
      setTrace(next);
      setStage("final_pages");
      setStatus("ready");
      onTrace(next);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Retrieval failed");
      setStatus("offline");
    } finally {
      setLoading(false);
    }
  }

  async function copyPrompt() {
    if (!trace) return;
    await navigator.clipboard.writeText(trace.copyable_prompt);
    setCopied(true);
  }

  return (
    <div className="live-console">
      <form className="console-form" onSubmit={submit}>
        <div className="console-form-head">
          <div>
            <span className="kicker">Live retrieval</span>
            <h3>Ask the frozen corpus</h3>
          </div>
          <span className={`api-status api-${status}`} aria-label={`API ${status}`}><i /> API {status}</span>
        </div>
        <label>
          Question
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about the ViDoRe V3 HR corpus"
            minLength={3}
            maxLength={2000}
            required
          />
        </label>
        <button
          className="example-query"
          type="button"
          onClick={() => setQuestion(BENCHMARK_EXAMPLE)}
        >
          Use a benchmark question
        </button>
        <div className="console-controls">
          <label>
            Retrieval mode
            <select value={mode} onChange={(event) => setMode(event.target.value as RetrievalMode)}>
              <option value="hybrid">Hybrid RRF</option>
              <option value="bm25">BM25 only</option>
              <option value="dense">Dense only</option>
            </select>
          </label>
          <label>
            Final pages
            <input type="number" min="1" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <label>
            Context budget
            <input type="number" min="64" max="4000" step="16" value={budget} onChange={(event) => setBudget(Number(event.target.value))} />
          </label>
        </div>
        <button className="button button-console" type="submit" disabled={loading || !question.trim()}>
          {loading ? "Running pipeline…" : "Run retrieval pipeline"}
        </button>
        {error && <div className="console-error"><strong>API unavailable</strong>{error}</div>}
      </form>

      {!trace ? (
        <div className="console-empty">
          <span>01 → 02 → 03 → 04</span>
          <h3>The trace appears here.</h3>
          <p>Inspect chunk rankings, page projection, rank fusion, budgeted evidence, and the exact prompt passed to an LLM.</p>
        </div>
      ) : (
        <div className="trace-output">
          <div className="trace-summary">
            <div><span>Total retrieval</span><strong>{trace.timings.total_ms.toFixed(1)} ms</strong></div>
            <div><span>Context</span><strong>{trace.context.used_tokens} / {trace.context.token_budget}</strong></div>
            <div><span>Evidence pages</span><strong>{trace.context.items.length}</strong></div>
            <div><span>Gold match</span><strong>{trace.benchmark_query_id ? "Yes" : "No"}</strong></div>
          </div>
          <div className="contract-strip">
            {trace.contract_checks.map((check) => (
              <div key={check.name} className={`contract-${check.status}`} title={check.detail}>
                <span>{check.status === "pass" ? "✓" : "!"}</span>{check.name}
              </div>
            ))}
          </div>
          <div className="trace-stage-tabs">
            {(Object.keys(STAGE_LABELS) as StageName[]).map((name) => (
              <button key={name} className={stage === name ? "active" : ""} onClick={() => setStage(name)}>
                {STAGE_LABELS[name]} <span>{trace.stages[name].length}</span>
              </button>
            ))}
          </div>
          <HitList hits={activeHits} />
          <div className="prompt-output">
            <div>
              <span className="kicker">LLM ready context</span>
              <h3>Copyable grounded prompt</h3>
              <p>The prompt treats documents as untrusted data and requires exact page IDs and quotes.</p>
            </div>
            <textarea readOnly value={trace.copyable_prompt} aria-label="Copyable LLM prompt" />
            <div className="prompt-actions">
              <button className="button button-dark" type="button" onClick={copyPrompt}>{copied ? "Copied" : "Copy prompt"}</button>
              <button className="button button-outline" type="button" onClick={onOpenValidator}>Validate an answer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
