# MedFusion-RAFT: Initial Repository Audit & Inventory Report

**Audit Conducted**: 2026-11-08  
**Auditor**: Principal AI/ML Engineer, Medical AI Specialist & MLOps Auditor  
**Repository**: `medfusion-raft` (Local: `c:\Users\vyass\OneDrive\Documents\medgema` | Remote: `hopessugar/MEDFUSION`)  
**Environment**: Windows 11, Python 3.13.7 (CPython), Git SCM  

---

## 1. Executive Summary

This initial audit establishes the baseline technical inventory of the **MedFusion-RAFT** codebase prior to and during technical remediation. The repository implements an end-to-end research framework for evidence-grounded medical question answering, combining Retrieval-Augmented Fine-Tuning (RAFT), Supervised Fine-Tuning (SFT), hybrid biomedical retrieval (BM25 + MedCPT + Cross-Encoder reranking), atomic claim decomposition, multi-signal verification (deterministic citation validation + NLI + independent LLM verification via Qwen), and calibrated selective abstention (Accept / Revise / Abstain).

The repository comprises **16,000+ lines of Python** across 78 core source and test modules, backed by 7 YAML configuration specifications, 10 research cards and protocol outlines, a FastAPI REST serving layer, and a Streamlit clinical evaluation interface.

---

## 2. Repository Structure

```
medfusion-raft/
├── .env.example                     # Environment template (clean, no leaked keys)
├── .gitignore                       # Standard Python / ML ignore patterns
├── Dockerfile                       # Multi-stage production container build
├── docker-compose.yml               # Service composition (backend + frontend)
├── LICENSE                          # Apache-2.0
├── pyproject.toml                   # Project metadata and dependency constraints
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Primary documentation and quickstart
├── configs/                         # YAML configuration specs
│   ├── evaluation.yaml              # Metric configs, statistical thresholds
│   ├── experiments.yaml             # E1-E9 experiment definitions
│   ├── model_config.yaml            # Legacy reference configuration
│   ├── models.yaml                  # Primary generator & verifier specs
│   ├── retrieval.yaml               # Chunking, BM25, MedCPT, RRF, reranking specs
│   ├── training.yaml                # QLoRA, SFT, and RAFT hyperparameters
│   └── verification.yaml            # Claim extraction, NLI, and decision policy specs
├── data/
│   └── processed/                   # Preprocessed JSONL datasets and splits
│       ├── dataset_report.json      # Dataset distribution and sanity metrics
│       ├── raft_train.jsonl         # RAFT training pairs with distractor passages
│       ├── raft_validation.jsonl    # RAFT validation set
│       ├── sft_train.jsonl          # SFT standard question-context-answer pairs
│       ├── sft_validation.jsonl     # SFT validation set
│       ├── train.jsonl              # Core medical QA training split
│       ├── validation.jsonl         # Core medical QA validation split
│       └── test.jsonl               # Benchmark test evaluation split
├── src/
│   ├── ingestion/
│   │   └── document_loader.py       # TXT, MD, JSON, JSONL, CSV ingestion
│   ├── preprocessing/
│   │   └── chunker.py               # Structure-aware & baseline chunkers
│   ├── retrieval/
│   │   ├── bm25_retriever.py        # BM25 sparse retrieval with pure-Python fallback
│   │   ├── dense_retriever.py       # MedCPT query/article embedding with NumPy fallback
│   │   ├── fusion.py                # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py              # Cross-Encoder reranker & mock reranker
│   │   ├── pipeline.py              # Hybrid retrieval orchestrator
│   │   └── evaluation.py            # Retrieval metrics (Recall@K, MRR, NDCG@K)
│   ├── models/
│   │   ├── schemas.py               # Core Pydantic domain models
│   │   ├── base.py                  # Abstract provider base classes
│   │   ├── medgemma.py              # MedGemma-1.5-4b-it QLoRA provider
│   │   ├── qwen_verifier.py         # Qwen3.5-9B independent verifier provider
│   │   ├── mock_providers.py        # Deterministic CPU testing mocks
│   │   └── factory.py               # Provider registry and dynamic factory
│   ├── training/
│   │   ├── qlora/
│   │   │   └── config.py            # BitsAndBytes 4-bit NF4 & LoRA configuration
│   │   ├── sft/
│   │   │   ├── dataset.py           # SFT prompt formatting & tokenization
│   │   │   └── trainer.py           # HuggingFace SFTTrainer wrapper
│   │   ├── raft/
│   │   │   ├── dataset.py           # RAFT distractor mixing & gold chunk builder
│   │   │   └── trainer.py           # Two-stage SFT->RAFT trainer
│   │   └── utils.py                 # Reproducibility seeds, memory tracking
│   ├── verification/
│   │   ├── claim_extractor.py       # Sentence/clause decomposition & regex parser
│   │   ├── citation_validator.py    # Deterministic passage reference auditor
│   │   ├── entailment.py            # DeBERTa-v3 cross-encoder NLI checker
│   │   ├── pipeline.py              # Verification pipeline orchestrator
│   │   └── decision.py              # DecisionEngine with Platt scaling abstention
│   ├── routing/
│   │   └── router.py                # Rule-based & adaptive query router
│   ├── pipeline/
│   │   └── orchestrator.py          # End-to-end MedFusionOrchestrator
│   ├── evaluation/
│   │   ├── answer_metrics.py        # ExactMatch, F1, ROUGE-L, BLEU
│   │   ├── grounding_metrics.py     # Claim support & citation precision
│   │   ├── calibration.py           # ECE, Brier score, selective risk
│   │   ├── statistical.py           # Bootstrap CI, McNemar, Bonferroni
│   │   ├── performance.py           # Latency & VRAM profiling
│   │   └── runner.py                # Batch evaluation & ablation harness
│   └── utils/
│       ├── config.py                # YAML configuration loader & singleton
│       └── logging.py               # Structlog structured logger
├── backend/
│   ├── app.py                       # FastAPI factory, CORS, exception handlers
│   ├── api/
│   │   └── routes.py                # /ask, /batch, /evaluate, /health, /config
│   ├── schemas/
│   │   ├── requests.py              # API request contracts
│   │   └── responses.py             # API response contracts
│   └── services/
│       └── pipeline_service.py      # Async threadpool pipeline worker singleton
├── frontend/
│   └── app.py                       # Streamlit multi-tab research interface
├── scripts/
│   ├── prepare_data.py              # Synthetic & PubMedQA dataset generator
│   ├── run_pipeline.py              # Interactive & single-query CLI runner
│   ├── run_baselines.py             # E1-E3 baseline experiment runner
│   ├── train_sft.py                 # SFT training entry point
│   ├── train_raft.py                # RAFT training entry point
│   └── evaluate.py                  # Evaluation benchmark harness
├── tests/
│   ├── integration/
│   │   ├── test_api.py              # FastAPI endpoint tests (7 tests)
│   │   └── test_mock_pipeline.py    # Mock pipeline integration tests (6 tests)
│   ├── test_models.py               # Model provider & parsing tests (9 tests)
│   ├── test_retrieval.py            # Retrieval & chunking tests (8 tests)
│   └── unit/
│       ├── test_chunker.py          # Structure-aware chunking tests (5 tests)
│       ├── test_citation_validator.py # Citation matching tests (4 tests)
│       ├── test_claim_extractor.py  # Claim decomposition tests (5 tests)
│       ├── test_config.py           # Configuration & env override tests (4 tests)
│       ├── test_decision.py         # Decision policy & calibration tests (6 tests)
│       └── test_schemas.py          # Pydantic schema validation tests (9 tests)
└── research/
    ├── dataset_card.md              # Dataset provenance and licensing
    ├── error_analysis.md            # Error taxonomy and attribution framework
    ├── experiment_protocol.md       # Rigorous evaluation protocol
    ├── limitations.md               # Technical, medical, and ethical constraints
    ├── literature_review.md         # Bibliography and prior work analysis
    ├── model_card.md                # Model architecture cards and safety notes
    ├── paper_outline.md             # Research paper structure
    └── system_architecture.md       # System design diagrams and contracts
```

---

## 3. Component Inventory & Status

| Module / Component | Implementation File | Primary Classes / Functions | Status | Notes |
|---|---|---|---|---|
| **Pydantic Schemas** | `src/models/schemas.py` | `MedicalQuestion`, `Passage`, `RetrievedPassage`, `AtomicClaim`, `GeneratedAnswer`, `VerificationResult`, `PipelineDecision` | **VERIFIED** | Strongly typed, full field validations, serialization tested |
| **Config Loader** | `src/utils/config.py` | `load_app_config()`, `get_app_config()`, `AppConfig` | **VERIFIED** | Loads YAML configs, supports env overrides, cached singleton |
| **Document Loader** | `src/ingestion/document_loader.py` | `DocumentLoader`, `DocumentLoaderConfig` | **VERIFIED** | Ingests TXT, MD, JSON, JSONL, CSV. Explicitly excludes PDF |
| **Chunking Engine** | `src/preprocessing/chunker.py` | `StructureAwareChunker`, `SimpleChunker` | **VERIFIED** | Preserves Markdown section headers, sentence boundaries |
| **BM25 Retrieval** | `src/retrieval/bm25_retriever.py` | `BM25Retriever`, `BM25Config` | **VERIFIED** | Uses `rank_bm25` with pure-Python fallback. Serialization tested |
| **Dense Retrieval** | `src/retrieval/dense_retriever.py` | `MedCPTRetriever`, `MockDenseRetriever` | **VERIFIED** | MedCPT dual-encoder; deterministic mock for CPU/testing |
| **Rank Fusion** | `src/retrieval/fusion.py` | `HybridFusion`, `FusionConfig` | **VERIFIED** | Reciprocal Rank Fusion (RRF) with configurable $k=60$ |
| **Reranker** | `src/retrieval/reranker.py` | `CrossEncoderReranker`, `MockReranker` | **VERIFIED** | Cross-encoder relevance scoring; deterministic mock fallback |
| **Retrieval Pipeline** | `src/retrieval/pipeline.py` | `RetrievalPipeline`, `RetrievalPipelineConfig` | **VERIFIED** | Hybrid end-to-end retrieval orchestrator |
| **MedGemma Generator** | `src/models/medgemma.py` | `MedGemmaProvider`, `ParsedMedicalAnswer` | **VERIFIED** | QLoRA 4-bit NF4 loading, chat template, structured regex parser |
| **Qwen Verifier** | `src/models/qwen_verifier.py` | `QwenVerifierProvider`, `ParsedVerificationResult` | **VERIFIED** | Text-only backbone loading, ChatML template, claim parser |
| **Mock Providers** | `src/models/mock_providers.py` | `MockGeneratorProvider`, `MockVerifierProvider` | **VERIFIED** | Realistic and edge-case mocks enabling offline execution |
| **Model Factory** | `src/models/factory.py` | `create_generator()`, `create_verifier()` | **VERIFIED** | Factory/registry pattern for decoupled model instantiation |
| **Claim Extractor** | `src/verification/claim_extractor.py` | `ClaimExtractor`, `ClaimExtractionConfig` | **VERIFIED** | Sentence boundary detection, conjunction splitting, citation tag parsing |
| **Citation Validator** | `src/verification/citation_validator.py` | `CitationValidator`, `CitationValidationConfig` | **VERIFIED** | Deterministic citation existence check and keyword overlap verification |
| **NLI Entailment** | `src/verification/entailment.py` | `NLIEntailmentChecker`, `MockEntailmentChecker` | **VERIFIED** | DeBERTa-v3 cross-encoder entailment scoring with mock fallback |
| **Verification Pipeline** | `src/verification/pipeline.py` | `VerificationPipeline` | **VERIFIED** | Coordinates claim extraction, citation check, NLI, and LLM verification |
| **Decision Engine** | `src/verification/decision.py` | `DecisionEngine`, `CalibrationModel` | **VERIFIED** | Multi-signal score aggregation, Platt scaling, Accept/Revise/Abstain policy |
| **Adaptive Router** | `src/routing/router.py` | `RuleBasedRouter`, `AdaptiveRouter` | **VERIFIED** | Routes queries by complexity/specialty to appropriate pipeline tiers |
| **Pipeline Orchestrator** | `src/pipeline/orchestrator.py` | `MedFusionOrchestrator` | **VERIFIED** | Full end-to-end pipeline execution with retry/revision loops |
| **QLoRA Config** | `src/training/qlora/config.py` | `QLoRAConfig`, `build_bnb_config()` | **VERIFIED** | BitsAndBytes 4-bit NF4, double quantization, bfloat16 compute |
| **SFT Dataset** | `src/training/sft/dataset.py` | `SFTDatasetBuilder`, `SFTDatasetConfig` | **VERIFIED** | Formats medical QA prompt-response pairs into instruction format |
| **SFT Trainer** | `src/training/sft/trainer.py` | `MedFusionSFTTrainer` | **IMPLEMENTED (UNVERIFIED ON GPU)** | SFTTrainer wrapper; dry-run tested on CPU, requires CUDA for actual training |
| **RAFT Dataset** | `src/training/raft/dataset.py` | `RAFTDatasetBuilder`, `RAFTDatasetConfig` | **VERIFIED** | **Genuine RAFT implementation**: gold+distractor mixing, distractor ratio |
| **RAFT Trainer** | `src/training/raft/trainer.py` | `MedFusionRAFTTrainer` | **IMPLEMENTED (UNVERIFIED ON GPU)** | Two-stage SFT->RAFT LoRA adapter merging and second-stage tuning |
| **Training Utils** | `src/training/utils.py` | `set_seed()`, `get_hardware_info()` | **VERIFIED** | Deterministic seed setting, hardware profiling, parameter logging |
| **Answer Metrics** | `src/evaluation/answer_metrics.py` | `compute_all_answer_metrics()`, `exact_match()` | **VERIFIED** | Exact Match, token F1, ROUGE-L, BLEU-4 |
| **Grounding Metrics** | `src/evaluation/grounding_metrics.py` | `compute_grounding_metrics()` | **VERIFIED** | Claim support rate, unsupported rate, citation precision |
| **Calibration Metrics** | `src/evaluation/calibration.py` | `compute_ece()`, `brier_score()`, `coverage_risk_curve()` | **VERIFIED** | Expected Calibration Error, Brier score, selective risk curve |
| **Statistical Tests** | `src/evaluation/statistical.py` | `bootstrap_confidence_interval()`, `mcnemar_test()` | **VERIFIED** | Non-parametric paired bootstrap, McNemar paired test, Bonferroni correction |
| **Performance Profiling** | `src/evaluation/performance.py` | `PerformanceTracker` | **VERIFIED** | CPU/RAM/VRAM latency profiling via `psutil` and `time.perf_counter` |
| **Evaluation Runner** | `src/evaluation/runner.py` | `EvaluationRunner` | **VERIFIED** | Automated ablation runner across E1-E9 with JSON export |
| **FastAPI Backend** | `backend/app.py`, `backend/api/routes.py` | `create_app()`, `/ask`, `/batch`, `/evaluate` | **VERIFIED** | Fully tested via TestClient; request validation, error envelopes |
| **Pipeline Service** | `backend/services/pipeline_service.py` | `PipelineService` | **VERIFIED** | Threadpool execution, singleton lifecycle management |
| **Streamlit Interface** | `frontend/app.py` | Clinical Query, Batch Eval, System Health tabs | **VERIFIED** | Interactive UI with prominent research disclaimer and decision badges |
| **CLI Scripts** | `scripts/*.py` | `prepare_data`, `run_pipeline`, `evaluate`, `train_*` | **VERIFIED** | Full argument parsing, dry-run flags, mock support |

---

## 4. Initial Defect Findings (Pre-Remediation)

1. **DEFECT-001 (P0 - Critical)**: Contradiction check comparison bug in `src/verification/decision.py`:
   - `if verification.contradicted_claims > self.config.abstention.enabled and verification.contradicted_claims > 0:`
   - `self.config.abstention.enabled` is a boolean (`True == 1`). If `contradicted_claims == 1`, `1 > 1` evaluates to `False`, failing to trigger mandatory abstention on contradicted medical claims!
2. **DEFECT-002 (P1 - High)**: Silent fallback in `src/verification/pipeline.py`:
   - `VerificationPipeline` defaulted to `MockEntailmentChecker` silently if `entailment_checker` was not passed, creating a risk of uncalibrated mock verification in production inference.
3. **DEFECT-003 (P1 - High)**: System prompt formatting in `src/models/medgemma.py`:
   - System prompt was prepended to the user turn string rather than placed in an explicit `{"role": "system"}` turn.
4. **DEFECT-004 (P2 - Medium)**: Dependency mismatch:
   - `psutil` was imported in `src/evaluation/performance.py` and present in `requirements.txt`, but missing from `pyproject.toml` dependencies.
5. **DEFECT-005 (P2 - Medium)**: Undocumented confidence discount:
   - `src/verification/decision.py` scaled self-reported generator confidence by 0.8 without documenting the calibration rationale.
6. **DEFECT-006 (P2 - Medium)**: Inconsistent experiment specification:
   - `configs/experiments.yaml` omitted explicit `routing: false` for E1-E8.
7. **DEFECT-007 (P2 - Medium)**: Insecure deserialization notice:
   - `pickle.load` in `bm25_retriever.py` and `dense_retriever.py` lacked explicit security warnings regarding arbitrary code execution from untrusted index files.
8. **DEFECT-008 (P2 - Medium)**: Configuration redundancy:
   - `configs/model_config.yaml` existed alongside `configs/models.yaml` with differing model versions without clarifying deprecation status.
