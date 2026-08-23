import { type FormEvent, useEffect, useState } from "react";
import { validateAnswer, type RetrievalTrace, type ValidationResult } from "./api";
import { percent } from "./format";

function templateFor(trace: RetrievalTrace | null): string {
  const pageId = trace?.context.items[0]?.page_id ?? "exact page_id from the evidence";
  return JSON.stringify(
    {
      answer: "",
      refused: false,
      citations: [{ page_id: pageId, quote: "", bounding_box: null }],
    },
    null,
    2,
  );
}

export default function OutputValidator({ trace }: { trace: RetrievalTrace | null }) {
  const [value, setValue] = useState(templateFor(trace));
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setValue(templateFor(trace));
    setResult(null);
    setError(null);
  }, [trace]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!trace) return;
    setLoading(true);
    setError(null);
    try {
      const parsed = JSON.parse(value) as unknown;
      setResult(await validateAnswer(trace.trace_id, parsed));
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : "Validation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="validator-shell">
      <form className="validator-form" onSubmit={submit}>
        <div>
          <span className="kicker">Grounding validator</span>
          <h3>Paste the LLM response</h3>
          <p>Validation uses the server retained trace, not evidence resubmitted by the browser.</p>
        </div>
        {!trace && <div className="validator-notice">Run the Query Console first. A validation is always bound to one exact retrieval trace.</div>}
        {trace && (
          <div className="trace-binding">
            <span>Bound trace</span>
            <code>{trace.trace_id}</code>
            <small>{trace.question}</small>
          </div>
        )}
        <label>
          GeneratedAnswer JSON
          <textarea value={value} onChange={(event) => setValue(event.target.value)} spellCheck={false} />
        </label>
        <button className="button button-console" type="submit" disabled={!trace || loading}>
          {loading ? "Validating…" : "Validate grounded output"}
        </button>
        {error && <div className="console-error"><strong>Validation rejected</strong>{error}</div>}
      </form>

      <div className="validation-output">
        {!result ? (
          <div className="console-empty">
            <span>Schema → pages → quotes → gold</span>
            <h3>Checks appear here.</h3>
            <p>Arbitrary questions receive structural and grounding checks. Exact benchmark questions also receive reference answer and gold evidence scores.</p>
          </div>
        ) : (
          <>
            <div className={`validation-verdict ${result.valid ? "valid" : "invalid"}`}>
              <span>{result.valid ? "✓" : "×"}</span>
              <div><small>Validation result</small><strong>{result.valid ? "Grounded contract passed" : "Grounded contract failed"}</strong></div>
            </div>
            <div className="validation-checks">
              {result.checks.map((check) => (
                <article key={check.name} className={`check-${check.status}`}>
                  <span>{check.status === "pass" ? "✓" : check.status === "warning" ? "!" : "×"}</span>
                  <div><strong>{check.name}</strong><p>{check.detail}</p></div>
                </article>
              ))}
            </div>
            <div className="grounding-metrics">
              <div><span>Context citations</span><strong>{percent(result.grounding.valid_context_precision)}</strong></div>
              <div><span>Quote support</span><strong>{percent(result.grounding.quote_support_precision)}</strong></div>
              <div><span>Gold page recall</span><strong>{result.benchmark.status === "applicable" ? percent(result.grounding.gold_page_recall) : "N/A"}</strong></div>
              <div><span>Reference F1</span><strong>{result.benchmark.answer_quality ? percent(result.benchmark.answer_quality.token_f1) : "N/A"}</strong></div>
            </div>
            <div className="benchmark-boundary">
              <strong>{result.benchmark.status === "applicable" ? "Benchmark scoring applied" : "Correctness not scored"}</strong>
              <p>{result.benchmark.status === "applicable" ? "The question exactly matched a ViDoRe query, so reference and gold evidence metrics are valid." : result.benchmark.reason}</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
