# ViDoRe RAG Beta

> A measurable visual document RAG prototype with paired retrieved and oracle generation, explicit evidence contracts, and honest failure attribution.

<div align="center">

`Rendered pages` → `Canonical records` → `Retrieval` → `Evidence` → `Generation` → `Evaluation`

**Local first · Reproducible · Evaluation aware · Fail closed**

</div>

## ✦ What exists today

| Capability | State | Proof |
|:--|:--:|:--|
| Typed dataset and evaluation contracts | ✅ | Invalid capability combinations are rejected |
| Canonical pages and chunks | ✅ | Every chunk retains document and page provenance |
| Deterministic text chunking | ✅ | Identical input creates identical chunk identities |
| Sparse retrieval with BM25 | ✅ | The example query finds the correct evidence page |
| Chunk to page projection | ✅ | A validated, fingerprinted projection contract precedes evaluation |
| Reciprocal Rank Fusion | ✅ | Sparse and dense page rankings are combined without score calibration |
| Artifact fingerprints | ✅ | Configuration and model revisions affect identity |
| Failure attribution | ✅ | Retrieval, context, generation, citation, and latency classification is wired and exercised by Gate C |
| Automated verification | ✅ | CMake, CTest, Ruff, strict mypy, CI, 61 Python tests, 7 frontend tests, and real-artifact regressions |
| ViDoRe V3 HR adapter | ✅ | Frozen revision, English queries, graded page qrels, answers, and pixel boxes |
| Cached OCR ingestion | ✅ | 1,110 pages processed with zero failures and fingerprinted manifests |
| Dense and hybrid retrieval | ✅ | Pinned MiniLM embeddings plus page level Reciprocal Rank Fusion |
| Structure aware context | ✅ | Headings, tables, figures, citations, and fixed token budgets |
| Real benchmark results | ✅ | All 318 English queries evaluated across five controlled runs |
| Paired RAG generation | ✅ | Retrieved and oracle paths, structured answers, cost, tokens, TTR, citations, resumable jobs, and live Gate C results |
| Interactive evidence lab | ✅ | 24 real queries with answer controls, cited source pages, human boxes, cost, tokens, and TTR |
| Live research workbench | ✅ | FastAPI retrieval traces, copyable grounded prompts, and trace-bound output validation |

## ◈ See it work

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,prototype,generation]'
```

Run the deterministic contract smoke test. This fixture is not used for benchmark claims:

```bash
PYTHONPATH=src python -m vidore_rag demo retrieve \
  --pages tests/fixtures/pages.json \
  --question "What does ERR-42 mean?" \
  --chunk-size 8 \
  --overlap 2
```

Expected first result:

```text
manual:2  →  Error ERR-42 means the battery temperature is above its safe operating range
```

The retriever searches chunks. The projection layer then converts those candidates into page evidence so evaluation happens at the same granularity as the dataset judgments.

## ◇ Explore the evidence lab

The deployable frontend replays the completed 24 query Gate C evaluation. It uses real answers, citations, source pages, human bounding boxes, token usage, cost, and latency measurements. It does not make paid model calls or expose an API key.

```bash
cd frontend
npm ci
npm run dev
```

Create the static production bundle:

```bash
npm test
npm run build
```

For a GitHub import in Vercel, set the Root Directory to `frontend`. The committed `vercel.json` builds and serves `dist`. Refresh the evidence bundle from the ignored local artifacts with `PYTHONPATH=src .venv/bin/python scripts/export_frontend_demo.py`.

### Run the live workbench

Install the API and retrieval dependencies:

```bash
python -m pip install -e '.[dev,api,prototype]'
```

Start FastAPI from the repository root:

```bash
PYTHONPATH=src uvicorn vidore_rag.api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

In a second terminal, point Vite at that API:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

The **Query Console** runs BM25, MiniLM dense retrieval, page projection, RRF, and token-budgeted context construction against the committed real corpus artifacts. It exposes each intermediate ranking and the exact prompt rather than substituting simulated data.

The **Output Validator** is bound to a server-retained retrieval trace. It checks the strict JSON schema, citation membership, exact quote support, and pixel coordinate bounds. If the question exactly matches one of the 318 benchmark questions, it also reports reference-answer and gold-evidence metrics. For arbitrary questions, correctness is explicitly `not_applicable` because no trustworthy gold answer exists.

No LLM provider is called by the public workbench. Users copy the grounded prompt to a provider they control, then paste the structured answer back for deterministic validation.

### Deploy the two services

The frontend remains a static Vercel deployment. Deploy `render.yaml` as a Render Blueprint for the read-only FastAPI service, then set this Vercel environment variable and redeploy the frontend:

```text
VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

The Blueprint deliberately selects Render Standard with 2 GB RAM. Render Free and Starter provide 512 MB, which is not a reliable memory envelope for the full PyTorch and Transformers hybrid path. This is a paid deployment decision; change it only if you also remove or replace the dense runtime.

The Render build downloads and verifies the exact pinned MiniLM revision against the committed embedding dimensions. Runtime network lookup is then disabled and the encoder is preloaded before the health check becomes ready. A missing model fails startup immediately instead of hanging the first hybrid request.

The API verifies every committed artifact checksum at startup, permits only declared CORS origins, validates request bodies, rate limits non-health requests, assigns request IDs, and emits structured request logs. The in-memory validation trace is intentionally bounded and expires after one hour. A multi-instance production deployment should replace that process-local trace store with Redis.

## ◎ Run the tests

Configure the test graph once:

```bash
cmake -S . -B build
```

Run every test declared in `testlist.list`:

```bash
ctest --test-dir build --output-on-failure
```

Run every local quality gate:

```bash
cmake --build build --target lint
cmake --build build --target typecheck
cmake --build build --target check
```

Each manifest entry becomes its own CTest test. Configuration fails if a listed file is missing or if a test file was added without updating the manifest. A failing module is named directly in CI output.

## ⌁ Architecture

```text
                         OFFLINE

  frozen ViDoRe adapter ──→ page images + records ──→ cached Tesseract OCR
             │                         │                         │
             │                         └──── shipped markdown ───┤
             │                                                   ▼
             └──────── manifests + fingerprints ──→ fixed / structured chunks
                                                               │
                                               ┌───────────────┴──────────────┐
                                               ▼                              ▼
                                          BM25 index                    MiniLM matrix

                         ONLINE

  question ──→ BM25 + dense retrieval ──→ page projection ──→ RRF
                                                                  │
                                      ▼
                         token-budgeted evidence ────────────────┐
                                      │                          │
                                      ▼                          ▼
                              retrieved answer             oracle answer
                                      │                          │
                                      └────────────┬─────────────┘
                                                   ▼
                                      answer / citation / cost / TTR
                                      first-failing-stage attribution
```

| Boundary | Responsibility |
|:--|:--|
| Dataset adapter | Translate source specific records into stable internal contracts |
| Canonical representation | Preserve identity, provenance, text, image references, and geometry |
| Chunker | Create deterministic retrieval units without losing page ownership |
| Retriever | Rank evidence candidates independently of evaluation semantics |
| Projection | Convert retrieval units into the dataset judgment unit |
| Evaluator | Score only metrics supported by declared dataset capabilities |

## ◉ Reproduce the real pipeline

The corpus is frozen to `vidore/vidore_v3_hr` revision `0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3`. Tesseract 5 must be available on the machine.

```bash
PYTHONPATH=src python -m vidore_rag dataset materialize
PYTHONPATH=src python -m vidore_rag ocr run --workers 8
PYTHONPATH=src python -m vidore_rag ocr compare

PYTHONPATH=src python -m vidore_rag index build \
  --policy structure \
  --text-source markdown

PYTHONPATH=src python -m vidore_rag index build \
  --policy fixed \
  --text-source markdown

PYTHONPATH=src python -m vidore_rag index dense
```

Run one shipped question with gold pages and a citation-ready context bundle:

```bash
PYTHONPATH=src python -m vidore_rag benchmark query \
  --query-id 0 \
  --mode hybrid \
  --context-budget 600
```

Run the complete benchmark while retaining query-level evidence in a JSON report:

```bash
PYTHONPATH=src python -m vidore_rag benchmark evaluate \
  --mode hybrid \
  --output results/hybrid.json \
  --summary-only
```

## ◌ Run paired generation

The committed configuration contains only the environment-variable name, never a secret. Export your key in the current shell:

```bash
export OPENAI_API_KEY="your-key"
```

Run one real ViDoRe question through retrieved evidence and gold oracle evidence with identical prompts and budgets:

```bash
PYTHONPATH=src python -m vidore_rag generation query \
  --query-id 0 \
  --mode hybrid
```

Run a small resumable experiment before spending on all 318 queries:

```bash
PYTHONPATH=src python -m vidore_rag generation run \
  --mode hybrid \
  --limit-queries 5
```

Run the declared representative Gate C slice:

```bash
PYTHONPATH=src python -m vidore_rag generation run \
  --mode hybrid \
  --selection configs/experiments/gate_c.yaml
```

Progress is printed per completed query pair. Final JSON remains machine-readable on standard output.

Every completed query is written atomically to its own file. Repeating the command reuses valid completed results and retries unfinished queries. The experiment fingerprint includes the dataset, text index, dense index, projection, provider, model, prompt version, token budgets, scoring threshold, and selected query IDs.

`configs/generation/openai.yaml` pins the current model and token prices. Update and commit that configuration whenever the provider changes pricing; cost is reported as an estimate derived from provider-reported usage.

## ✧ Measured results

Every row uses the same 318 English queries, page-level graded judgments, `top_k = 10`, and local Apple Silicon machine. Latency covers warm retrieval, projection, and fusion; it excludes one-time model loading.

| Text | Retrieval | nDCG@10 | Recall@10 | MRR | p50 ms | p95 ms |
|:--|:--|--:|--:|--:|--:|--:|
| Shipped markdown, fixed chunks | BM25 | 0.4865 | 0.5322 | 0.6113 | 16.79 | 23.61 |
| Shipped markdown, fixed chunks | MiniLM dense | 0.4400 | 0.4997 | 0.5636 | 6.71 | 18.42 |
| Shipped markdown, fixed chunks | BM25 + dense RRF | **0.5194** | **0.5792** | **0.6214** | 29.21 | 41.35 |
| Shipped markdown, structured chunks | BM25 | 0.4742 | 0.5305 | 0.5964 | 19.40 | 26.91 |
| Tesseract OCR, fixed chunks | BM25 | 0.4719 | 0.5162 | 0.5990 | 16.78 | 23.89 |

RRF improves nDCG@10 by 6.8% and Recall@10 by 8.8% relative to the strongest single retriever. The first structure-aware heuristic does not beat fixed chunks, which is a useful failure result: preserving block boundaries alone is insufficient without better layout recovery and query-aware table handling.

Gold evidence tags reveal where this text-only retrieval stack struggles. Membership is multi-label, so one query may contribute to several rows.

| Gold evidence type | n queries | nDCG@10 | Recall@10 |
|:--|--:|--:|--:|
| Text | 310 | 0.5231 | 0.5844 |
| Table | 112 | 0.4427 | 0.4783 |
| Chart | 104 | 0.4962 | 0.5090 |
| Infographic | 33 | 0.5395 | 0.3956 |

Tables trail Text by 0.0805 absolute nDCG@10. Infographics do not underperform on nDCG, but their Recall@10 is 0.1888 below Text. The result supports testing a targeted visual arm for tables and missed infographic evidence; it does not support claiming that every visual category is uniformly worse. Reproduce the table with `benchmark evidence-breakdown`; the machine-readable result is in `examples/evidence_type_breakdown.json`.

The five-query Gate B generation run completed all ten retrieved/oracle calls for `$0.024642` estimated total cost. Retrieved versus oracle mean token F1 was `0.4237 / 0.3849`; p50 TTR was `1.671s / 1.434s`; p95 TTR was `2.591s / 1.584s`. Because all five queries have multi-page gold with unknown AND/OR semantics, their stage attribution is reported as `unscorable_gold`. See `examples/gate_b_generation_summary.json`.

The deterministic 24-query Gate C run completed all 48 model calls with zero provider failures for `$0.118252`. Retrieved versus oracle mean token F1 was `0.4518 / 0.4884`; gold-page citation recall was `0.2495 / 0.5435`; p50 TTR was `1.512s / 1.293s`. Valid-context precision was `1.0` in both arms. The oracle gain isolates a retrieval contribution, while the remaining oracle errors show that better pages alone do not solve context interpretation and answer generation. See `examples/gate_c_generation_summary.json`.

Gate C also uses mutually exclusive evidence strata, six queries each. These small groups are diagnostic rather than statistically powered.

| Gold evidence stratum | Retrieved token F1 | Oracle token F1 | Retrieved gold-page citation recall | Retrieved p50 TTR |
|:--|--:|--:|--:|--:|
| Text-only | 0.4318 | 0.4565 | 0.1518 | 1.447s |
| Table | 0.5024 | 0.5685 | 0.2667 | 1.455s |
| Chart | 0.5283 | 0.5440 | 0.2198 | 1.523s |
| Infographic | 0.3448 | 0.3846 | 0.3596 | 2.416s |

Infographics produced the lowest answer F1 and slowest response in both arms. Their weak oracle result suggests that the next visual experiment must improve evidence interpretation and serialization, not merely page retrieval. Reproduce the table with `generation evidence-breakdown`; the machine-readable result is in `examples/gate_c_evidence_breakdown.json`.

The fixed-versus-structure-aware experiment held the Gate C query set, hybrid retriever, model, prompt, and token budgets constant. Structure-aware context changed 22 of 24 retrieved bundles. Retrieved token F1 moved from `0.4518` to `0.4642` (`+0.0123`), but the repeated identical-prompt oracle control moved from `0.4884` to `0.5031` (`+0.0148`). Subtracting that run drift gives a diagnostic adjusted delta of `-0.0024`. Gold-page citation recall improved only `0.0091`, quote support was flat, and full-corpus retrieval was effectively unchanged (`0.5194 → 0.5206` nDCG@10; `0.5792 → 0.5766` Recall@10). The structure heuristic therefore does not earn promotion over fixed chunks. Its lower measured cost is not treated as an architecture gain because the second run received provider prompt-cache discounts. See `examples/gate_c_fixed_vs_structure_generation.json`.

[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is intentionally a low-cost dense baseline, not evidence that dense retrieval generally underperforms sparse retrieval. It produces 384-dimensional embeddings, has six transformer layers, and truncates inputs beyond 256 word pieces. The [official ViDoRe V3 monolingual table](https://arxiv.org/html/2601.08620v2) reports HR BM25S nDCG@10 of `0.496`; this implementation's English-only `0.4865` is directionally consistent despite different BM25 and chunking details.

The OCR diagnostic compared token overlap with shipped markdown across all pages: precision `0.8564`, recall `0.9461`, and F1 `0.8990`. This is a consistency signal, not a formal OCR ground-truth score.

## ◇ The honesty boundary

The selected contract is `vidore/vidore_v3_hr` at revision `0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3`. Inspect it with:

```bash
PYTHONPATH=src python -m vidore_rag dataset inspect \
  --config configs/datasets/vidore_v3.yaml
```

The command exits successfully only when the configuration is schema-valid and its adapter is registered. It prints those states separately; source availability is ultimately verified during materialization. The English slice contains 318 populated free text answers and graded page judgments with pixel bounding boxes. The data does not explicitly declare whether multiple relevant pages are alternatives or jointly required, so that semantic remains `unknown` and the system will not fabricate multi hop attribution from it.

<details>
<summary><strong>Why retrieval units and judgment units are separate</strong></summary>

A retriever may search chunks while a benchmark labels pages. Scoring chunk identifiers directly against page identifiers creates meaningless metrics. Each index now carries a chunk-to-page mapping contract and projection fingerprint; the benchmark validates that contract against the dataset's declared judgment unit before evaluation.

</details>

<details>
<summary><strong>Why the first chunker is intentionally simple</strong></summary>

Fixed token windows provide a controlled baseline. The next context experiment will compare that baseline with structure aware handling for tables, figures, captions, and headings while holding retrieval candidates and token budgets constant.

</details>

## ⟡ Build path

| Stage | Deliverable | State |
|:--:|:--|:--:|
| 0 | Contracts, fingerprints, CLI, CI, and synthetic sparse slice | ✅ |
| 1 | Verified ViDoRe adapter, OCR, caching, and retrieval evaluation | ✅ |
| 2 | Dense retrieval plus measured RRF comparison | ✅ |
| 3 | Fixed context versus structure aware context experiment | Complete; no demonstrated structure-aware gain |
| 4 | Retrieved context versus oracle context generation | Gate C complete: 24 queries and 48 live calls |
| 5 | Results, failure analysis, one pager, and submission polish | Planned |

## ⊙ Scope boundary

| V3 Beta implementation | V3 design only |
|:--|:--|
| One verified ViDoRe slice | Multiple production dataset adapters |
| OCR text path | ColPali visual late interaction |
| Sparse, dense, and RRF comparison | Learned reranking |
| Fixed and structure aware context | Full native Office fidelity |
| Local Python application | Distributed services and Kubernetes |
| Measured retrieval quality and latency | Production deployment and autoscaling |
| Paired generation engine and 24-query stratified live measurements | Full 318-query generation benchmark |

## → Next move

Finish the one-pager and failure analysis using the measured Gate C and chunk-policy evidence. Keep the full 318-query paid run gated: the next technically informative experiment is a targeted visual arm, not more calls through either unchanged text-only chunker.
