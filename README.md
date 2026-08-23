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
| Failure attribution | ◇ | Retrieval, context, generation, citation, and latency classification is wired; live-provider validation remains |
| Automated verification | ✅ | CMake, CTest, Ruff, strict mypy, CI, 56 portable tests, and 2 real-artifact regressions |
| ViDoRe V3 HR adapter | ✅ | Frozen revision, English queries, graded page qrels, answers, and pixel boxes |
| Cached OCR ingestion | ✅ | 1,110 pages processed with zero failures and fingerprinted manifests |
| Dense and hybrid retrieval | ✅ | Pinned MiniLM embeddings plus page level Reciprocal Rank Fusion |
| Structure aware context | ✅ | Headings, tables, figures, citations, and fixed token budgets |
| Real benchmark results | ✅ | All 318 English queries evaluated across five controlled runs |
| Paired RAG generation | ◇ | Retrieved and oracle paths, structured answers, cost, tokens, TTR, citations, and resumable jobs are implemented; live results remain |

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
| 3 | Fixed context versus structure aware context experiment | Retrieval complete; live generation comparison pending |
| 4 | Retrieved context versus oracle context generation | Five-query live Gate B complete; Gate C prepared |
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
| Paired generation engine and five-query live measurements | Larger generation benchmark |

## → Next move

Run the fingerprint-bound 24-query Gate C selection, inspect the evidence-type strata, and scale only if quality, citations, cost, and TTR justify it. The V3 architecture remains broader than this intentionally tightened V3 Beta implementation.
