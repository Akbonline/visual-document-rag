# ViDoRe RAG Beta

> A measurable visual document retrieval prototype built to answer one question honestly: when an answer is wrong, which stage failed?

<div align="center">

`Rendered pages` → `Canonical records` → `Chunks` → `Retrieval` → `Page evidence` → `Evaluation`

**Local first · Reproducible · Evaluation aware · Fail closed**

</div>

## ✦ What exists today

| Capability | State | Proof |
|:--|:--:|:--|
| Typed dataset and evaluation contracts | ✅ | Invalid capability combinations are rejected |
| Canonical pages and chunks | ✅ | Every chunk retains document and page provenance |
| Deterministic text chunking | ✅ | Identical input creates identical chunk identities |
| Sparse retrieval with BM25 | ✅ | The example query finds the correct evidence page |
| Chunk to page projection | ✅ | Retrieval output matches page level judgments |
| Reciprocal Rank Fusion | ✅ | Ready to combine sparse and dense rankings |
| Artifact fingerprints | ✅ | Configuration and model revisions affect identity |
| Failure attribution | ✅ | Retrieval and generation failures remain distinct |
| Automated verification | ✅ | CMake, CTest, Ruff, mypy, CI, and 19 tests |
| ViDoRe V3 HR adapter | ✅ | Frozen revision, English queries, graded page qrels, answers, and pixel boxes |
| ViDoRe V3 OCR and measured results | ◌ | Adapter exists; corpus materialization and OCR are next |

## ◈ See it work

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the synthetic retrieval journey:

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

  dataset adapter
        │
        ▼
  canonical pages ──→ deterministic chunks ──→ retrieval index
        │                       │                       │
        └──────── provenance ───┴──── fingerprints ────┘

                         ONLINE

  question ──→ retriever ──→ chunk candidates ──→ page projection
                                                        │
                                                        ▼
                                               evidence for scoring
```

| Boundary | Responsibility |
|:--|:--|
| Dataset adapter | Translate source specific records into stable internal contracts |
| Canonical representation | Preserve identity, provenance, text, image references, and geometry |
| Chunker | Create deterministic retrieval units without losing page ownership |
| Retriever | Rank evidence candidates independently of evaluation semantics |
| Projection | Convert retrieval units into the dataset judgment unit |
| Evaluator | Score only metrics supported by declared dataset capabilities |

## ◇ The honesty boundary

The selected contract is `vidore/vidore_v3_hr` at revision `0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3`. Inspect it with:

```bash
PYTHONPATH=src python -m vidore_rag dataset inspect \
  --config configs/datasets/vidore_v3.yaml
```

The command exits successfully and prints the frozen capabilities. The English slice contains 318 populated free text answers and graded page judgments with pixel bounding boxes. The data does not explicitly declare whether multiple relevant pages are alternatives or jointly required, so that semantic remains `unknown` and the system will not fabricate multi hop attribution from it.

<details>
<summary><strong>Why retrieval units and judgment units are separate</strong></summary>

A retriever may search chunks while a benchmark labels pages. Scoring chunk identifiers directly against page identifiers creates meaningless metrics. This project requires an explicit, fingerprinted projection between those units before evaluation can run.

</details>

<details>
<summary><strong>Why the first chunker is intentionally simple</strong></summary>

Fixed token windows provide a controlled baseline. The next context experiment will compare that baseline with structure aware handling for tables, figures, captions, and headings while holding retrieval candidates and token budgets constant.

</details>

## ⟡ Build path

| Stage | Deliverable | State |
|:--:|:--|:--:|
| 0 | Contracts, fingerprints, CLI, CI, and synthetic sparse slice | ✅ |
| 1 | Verified ViDoRe adapter, OCR, caching, and official retrieval evaluation | Adapter complete; OCR next |
| 2 | Dense retrieval plus measured RRF comparison | Planned |
| 3 | Fixed context versus structure aware context experiment | Planned |
| 4 | Retrieved context versus oracle context generation | Planned |
| 5 | Results, failure analysis, one pager, and submission polish | Planned |

## ⊙ Scope boundary

| V3 Beta implementation | V3 design only |
|:--|:--|
| One verified ViDoRe slice | Multiple production dataset adapters |
| OCR text path | ColPali visual late interaction |
| Sparse, dense, and RRF comparison | Learned reranking |
| Fixed and structure aware context | Full native Office fidelity |
| Local Python application | Distributed services and Kubernetes |
| Measured latency, tokens, and cost | Production deployment and autoscaling |

## → Next move

Materialize the frozen ViDoRe V3 HR corpus, run OCR with cache fingerprints, and compare OCR text with shipped markdown before building the first measured sparse retrieval baseline.
