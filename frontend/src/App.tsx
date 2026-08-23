import { useEffect, useMemo, useState } from "react";
import rawData from "./data/demo.json";
import { compactNumber, dollars, percent, seconds } from "./format";
import type { Arm, DemoData, DemoQuery, Page } from "./types";

const data = rawData as DemoData;
type ArmName = "retrieved" | "oracle";

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M4 10h11M11 5l5 5-5 5" />
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.86c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.35 1.09 2.92.83.09-.65.35-1.09.64-1.34-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.6 9.6 0 0 1 12 6.84a9.6 9.6 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86V21c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
    </svg>
  );
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}

function Hero() {
  const generation = data.summaries.generation;
  const hybrid = data.summaries.retrieval.find((row) => row.label === "Hybrid RRF")!;
  const costPerAnswer = generation.retrieved.estimated_cost_usd / generation.query_count;
  return (
    <header className="hero" id="top">
      <nav className="nav shell">
        <a className="brand" href="#top" aria-label="Evidence Lab home">
          <span className="brand-mark"><DatabaseIcon /></span>
          <span>Evidence Lab</span>
        </a>
        <div className="nav-links">
          <a href="#experiment">Experiment</a>
          <a href="#results">Results</a>
          <a href="#reasoning">Reasoning</a>
          <a className="github-link" href="https://github.com/Akbonline/visual-document-rag" target="_blank" rel="noreferrer">
            <GitHubIcon /> Repository
          </a>
        </div>
      </nav>

      <div className="hero-grid shell">
        <div className="hero-copy">
          <div className="eyebrow"><span /> Visual document RAG, measured end to end</div>
          <h1>A document system that <em>shows its work.</em></h1>
          <p className="hero-lede">
            An inspectable OCR first pipeline for retrieving, answering, and citing evidence from complex documents. Every result below comes from a real ViDoRe V3 evaluation run.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href="#experiment">Inspect the evidence <ArrowIcon /></a>
            <a className="button button-secondary" href="#reasoning">Read the reasoning</a>
          </div>
        </div>
        <div className="hero-proof" aria-label="Measured system summary">
          <div className="proof-head">
            <span>Measured system · {data.dataset}</span>
            <Badge tone="green">318 queries</Badge>
          </div>
          <div className="proof-grid">
            <Metric label="Hybrid nDCG@10" value={hybrid.ndcg_at_10.toFixed(4)} note="6.8% over BM25" />
            <Metric label="Retrieval p95" value={`${hybrid.latency_p95_ms.toFixed(1)} ms`} note="300 ms budget" />
            <Metric label="Answer TTR p50" value={seconds(generation.retrieved.ttr_p50_ms)} note="6 second budget" />
            <Metric label="Cost per answer" value={dollars(costPerAnswer)} note="$2.63 per 1,000" />
          </div>
          <div className="proof-foot">
            <span className="pulse" /> Real pages, real model calls, reproducible artifacts
          </div>
        </div>
      </div>
    </header>
  );
}

function QueryList({
  selected,
  onSelect,
}: {
  selected: DemoQuery;
  onSelect: (query: DemoQuery) => void;
}) {
  const [filter, setFilter] = useState("All");
  const filters = ["All", "Text", "Table", "Chart", "Infographic"];
  const visible = data.queries.filter((query) =>
    filter === "All" ? true : query.evidenceTypes.includes(filter),
  );
  return (
    <aside className="query-panel">
      <div className="panel-title">
        <div>
          <span className="kicker">Gate C sample</span>
          <h3>24 real questions</h3>
        </div>
        <Badge tone="blue">Recorded</Badge>
      </div>
      <div className="filters" aria-label="Filter questions by evidence type">
        {filters.map((item) => (
          <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>
            {item}
          </button>
        ))}
      </div>
      <div className="query-list">
        {visible.map((query) => (
          <button
            key={query.queryId}
            className={`query-item ${selected.queryId === query.queryId ? "selected" : ""}`}
            onClick={() => onSelect(query)}
          >
            <span className="query-number">Q{query.nativeQueryId}</span>
            <span className="query-copy">
              <strong>{query.question}</strong>
              <small>{query.evidenceTypes.join(" · ")}</small>
            </span>
            <ArrowIcon />
          </button>
        ))}
      </div>
    </aside>
  );
}

function EvidenceViewer({ query, arm }: { query: DemoQuery; arm: Arm }) {
  const pageIds = useMemo(() => {
    const ids = [
      ...query.goldEvidence.map((item) => item.pageId),
      ...arm.answer.citations.map((citation) => citation.page_id),
    ];
    return [...new Set(ids)].filter((pageId) => data.pages[pageId]);
  }, [query, arm]);
  const [selectedPageId, setSelectedPageId] = useState(pageIds[0]);

  useEffect(() => {
    setSelectedPageId(pageIds[0]);
  }, [query.queryId, arm, pageIds]);

  const page = data.pages[selectedPageId] as Page | undefined;
  const gold = query.goldEvidence.find((item) => item.pageId === selectedPageId);
  const boxes = gold?.boundingBoxes ?? [];
  const citations = arm.answer.citations.filter((item) => item.page_id === selectedPageId);
  if (!page) return null;

  return (
    <div className="evidence-viewer">
      <div className="source-tabs" role="tablist" aria-label="Evidence pages">
        {pageIds.map((pageId) => {
          const candidate = data.pages[pageId];
          const isGold = query.goldEvidence.some((item) => item.pageId === pageId);
          const isCited = arm.answer.citations.some((item) => item.page_id === pageId);
          return (
            <button
              key={pageId}
              className={selectedPageId === pageId ? "active" : ""}
              onClick={() => setSelectedPageId(pageId)}
            >
              Page {candidate.pageNumber}
              <span>{isGold ? "Gold" : isCited ? "Cited" : "Context"}</span>
            </button>
          );
        })}
      </div>
      <div className="page-toolbar">
        <div>
          <strong>{page.documentTitle}</strong>
          <span>Page {page.pageNumber}</span>
        </div>
        <div className="legend">
          {gold && <span><i className="legend-box" /> Human evidence box</span>}
          {citations.length > 0 && <span><i className="legend-dot" /> Model cited</span>}
        </div>
      </div>
      <div className="page-stage">
        <div className="page-canvas">
          <img src={page.imageUrl} alt={`${page.documentTitle}, page ${page.pageNumber}`} />
          {boxes.map((box, index) => (
            <span
              className="gold-box"
              key={`${box.annotator ?? 0}-${index}`}
              style={{
                left: `${(box.x1 / page.width) * 100}%`,
                top: `${(box.y1 / page.height) * 100}%`,
                width: `${((box.x2 - box.x1) / page.width) * 100}%`,
                height: `${((box.y2 - box.y1) / page.height) * 100}%`,
              }}
              title={`Human evidence annotation ${index + 1}`}
            />
          ))}
        </div>
      </div>
      <div className="page-caption">
        {citations.length ? (
          citations.map((citation, index) => (
            <blockquote key={`${citation.page_id}-${index}`}>
              <span>Model citation {index + 1}</span>
              “{citation.quote}”
            </blockquote>
          ))
        ) : (
          <p>This is a human labeled gold page that the selected answer did not cite.</p>
        )}
      </div>
    </div>
  );
}

function AnswerPanel({ query }: { query: DemoQuery }) {
  const [armName, setArmName] = useState<ArmName>("retrieved");
  useEffect(() => setArmName("retrieved"), [query.queryId]);
  const arm = query[armName];
  return (
    <main className="answer-panel">
      <div className="answer-head">
        <div className="answer-meta">
          <span>Query {query.nativeQueryId}</span>
          {query.evidenceTypes.map((type) => <Badge key={type}>{type}</Badge>)}
        </div>
        <h2>{query.question}</h2>
        <p className="reference"><span>Reference answer</span>{query.referenceAnswers[0]}</p>
      </div>

      <div className="arm-tabs" role="tablist" aria-label="Answer context arm">
        <button className={armName === "retrieved" ? "active" : ""} onClick={() => setArmName("retrieved")}>
          Retrieved evidence
          <small>What the system found</small>
        </button>
        <button className={armName === "oracle" ? "active" : ""} onClick={() => setArmName("oracle")}>
          Oracle evidence
          <small>Gold pages as a control</small>
        </button>
      </div>

      <section className="generated-answer">
        <div className="section-label">
          <span>{armName === "retrieved" ? "System answer" : "Controlled answer"}</span>
          <Badge tone={arm.answer.refused ? "amber" : "green"}>{arm.answer.refused ? "Refused" : "Answered"}</Badge>
        </div>
        <p>{arm.answer.answer}</p>
      </section>

      <div className="answer-metrics">
        <Metric label="Token F1" value={percent(arm.answer_quality.token_f1)} />
        <Metric label="Gold citation recall" value={percent(arm.citation_quality.gold_page_recall)} />
        <Metric label="Quote support" value={percent(arm.citation_quality.quote_support_precision)} />
        <Metric label="TTR" value={seconds(arm.time_to_response_ms)} />
        <Metric label="Tokens" value={compactNumber(arm.usage.total_tokens)} />
        <Metric label="Cost" value={dollars(arm.cost.estimated_usd)} />
      </div>

      <div className="inspection-note">
        <strong>Why compare both?</strong>
        <span>The oracle arm holds the prompt and budget constant while replacing retrieved pages with human labeled pages. The gap helps separate retrieval failures from context and generation failures.</span>
      </div>

      <EvidenceViewer query={query} arm={arm} />
    </main>
  );
}

function Experiment() {
  const initial = data.queries.find((query) => query.nativeQueryId === 6) ?? data.queries[0];
  const [selected, setSelected] = useState(initial);
  return (
    <section className="experiment-section" id="experiment">
      <div className="section-intro shell">
        <div>
          <span className="kicker">Inspect one result end to end</span>
          <h2>Evidence, not just an answer.</h2>
        </div>
        <p>{data.recordingNotice}</p>
      </div>
      <div className="lab-shell shell">
        <QueryList selected={selected} onSelect={setSelected} />
        <AnswerPanel query={selected} />
      </div>
    </section>
  );
}

function Results() {
  const generation = data.summaries.generation;
  const maxNdcg = Math.max(...data.summaries.retrieval.map((row) => row.ndcg_at_10));
  return (
    <section className="results-section" id="results">
      <div className="shell">
        <div className="section-intro light">
          <div>
            <span className="kicker">Measured tradeoffs</span>
            <h2>The baseline earned each addition.</h2>
          </div>
          <p>All retrieval rows use the same 318 English queries and page level judgments. Generation uses a deterministic 24 query evidence stratified slice.</p>
        </div>
        <div className="results-grid">
          <article className="result-card retrieval-card">
            <div className="card-heading">
              <div><span>Retrieval</span><h3>nDCG@10</h3></div>
              <Badge tone="blue">318 queries</Badge>
            </div>
            <div className="bar-chart">
              {data.summaries.retrieval.map((row) => (
                <div className="bar-row" key={`${row.label}-${row.source}`}>
                  <div className="bar-label"><strong>{row.label}</strong><span>{row.source}</span></div>
                  <div className="bar-track"><span style={{ width: `${(row.ndcg_at_10 / maxNdcg) * 100}%` }} /></div>
                  <strong className="bar-value">{row.ndcg_at_10.toFixed(4)}</strong>
                </div>
              ))}
            </div>
            <p className="card-insight"><strong>Why hybrid:</strong> Reciprocal Rank Fusion raised nDCG@10 by 6.8% and Recall@10 by 8.8% over BM25 without requiring incomparable score calibration.</p>
          </article>

          <article className="result-card generation-card">
            <div className="card-heading">
              <div><span>Generation control</span><h3>Retrieved vs oracle</h3></div>
              <Badge tone="green">48 calls</Badge>
            </div>
            <div className="comparison-head"><span>Metric</span><span>Retrieved</span><span>Oracle</span></div>
            <div className="comparison-row"><span>Answer token F1</span><strong>{percent(generation.retrieved.mean_answer_token_f1)}</strong><strong>{percent(generation.oracle.mean_answer_token_f1)}</strong></div>
            <div className="comparison-row"><span>Gold citation recall</span><strong>{percent(generation.retrieved.mean_gold_page_recall)}</strong><strong>{percent(generation.oracle.mean_gold_page_recall)}</strong></div>
            <div className="comparison-row"><span>Median TTR</span><strong>{seconds(generation.retrieved.ttr_p50_ms)}</strong><strong>{seconds(generation.oracle.ttr_p50_ms)}</strong></div>
            <div className="comparison-row"><span>Estimated cost</span><strong>{dollars(generation.retrieved.estimated_cost_usd)}</strong><strong>{dollars(generation.oracle.estimated_cost_usd)}</strong></div>
            <p className="card-insight"><strong>Why the control matters:</strong> Better pages improved average answer F1, but oracle answers still failed. Retrieval alone was not the full bottleneck.</p>
          </article>
        </div>
      </div>
    </section>
  );
}

function Reasoning() {
  return (
    <section className="reasoning-section" id="reasoning">
      <div className="shell">
        <div className="section-intro">
          <div><span className="kicker">Architecture through evidence</span><h2>Why this system looks this way.</h2></div>
          <p>Four design iterations progressively replaced assumptions with explicit contracts, controlled experiments, and honest scope boundaries.</p>
        </div>
        <div className="pipeline" aria-label="System pipeline">
          {["ViDoRe adapter", "OCR + canonical IR", "BM25 + MiniLM", "Page projection", "RRF + context", "Answer + oracle", "Failure attribution"].map((step, index) => (
            <div className="pipeline-step" key={step}><span>{String(index + 1).padStart(2, "0")}</span><strong>{step}</strong>{index < 6 && <ArrowIcon />}</div>
          ))}
        </div>
        <div className="decision-grid">
          <article><span className="decision-number">01</span><h3>OCR first</h3><p><strong>Why:</strong> It keeps exact text, identifiers, coordinates, lexical search, and quote citations inspectable on CPU. Visual retrieval must beat this measured baseline before earning greater storage and hardware cost.</p></article>
          <article><span className="decision-number">02</span><h3>Hybrid retrieval</h3><p><strong>Why:</strong> BM25 wins on exact terms while dense retrieval captures paraphrase. Rank fusion combines those complementary signals without pretending their raw scores share a scale.</p></article>
          <article><span className="decision-number">03</span><h3>Page level contract</h3><p><strong>Why:</strong> Search operates on chunks, but ViDoRe judges pages. A fingerprinted projection makes that unit conversion explicit and prevents invalid evaluation comparisons.</p></article>
          <article><span className="decision-number">04</span><h3>Paired generation</h3><p><strong>Why:</strong> Retrieved and oracle arms use identical prompts and token budgets. That control distinguishes missing evidence from poor evidence interpretation or generation.</p></article>
        </div>
        <div className="honesty">
          <div><span className="kicker">Honesty boundary</span><h3>What this demo does not claim</h3></div>
          <p>It does not claim visual localization from a text only path, fully resolved multi page AND or OR semantics, or production deployment. ColPali, learned reranking, native Office ingestion, and a second benchmark adapter remain designed extensions, not completed features.</p>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  return (
    <>
      <Hero />
      <Experiment />
      <Results />
      <Reasoning />
      <footer>
        <div className="shell">
          <div className="brand"><span className="brand-mark"><DatabaseIcon /></span><span>Evidence Lab</span></div>
          <p>Built by Akshat Bajpai. ViDoRe V3 HR dataset, CC BY 4.0.</p>
          <a href="#top">Back to top ↑</a>
        </div>
      </footer>
    </>
  );
}
