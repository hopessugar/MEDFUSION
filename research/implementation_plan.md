# MedFusion-RAFT Implementation Plan

## Project Overview

**MedFusion-RAFT**: An Adaptive, Evidence-Grounded Medical Question Answering Framework with Retrieval-Augmented Fine-Tuning, Independent Verification and Selective Abstention.

## Repository Audit Results

- **Starting state**: Empty repository (greenfield)
- **Python environment**: Python 3.13.7, pip 25.3
- **OS**: Windows 11

## Model Availability Verification

| Model | Repository | Status | Architecture | Notes |
|-------|-----------|--------|-------------|-------|
| MedGemma 1.5-4b-it | `google/medgemma-1.5-4b-it` | ✅ Available (gated) | Gemma 3 base, 4B params | Requires HF token, BF16 native, QLoRA compatible |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | ✅ Available | Hybrid GatedDeltaNet+Attention VLM | Load as CausalLM for text-only; ChatML template |
| MedCPT Query Encoder | `ncbi/MedCPT-Query-Encoder` | ✅ Available | BERT-based encoder | Public, no gating |
| MedCPT Article Encoder | `ncbi/MedCPT-Article-Encoder` | ✅ Available | BERT-based encoder | Public, no gating |
| MedCPT Cross-Encoder | `ncbi/MedCPT-Cross-Encoder` | ✅ Available | BERT-based cross-encoder | AutoModelForSequenceClassification |

## Architecture Decision: Qwen3.5-9B as Verifier

Qwen3.5-9B is natively a VLM but can be loaded text-only via `AutoModelForCausalLM`, which strips the vision encoder and reduces VRAM. The text backbone uses a hybrid Gated Delta Networks + Sparse Attention architecture with 32 layers, 4096 hidden dim, and 262K context support. ChatML template (`<|im_start|>user`).

This satisfies the requirement for a different model family (Alibaba Qwen vs Google Gemma).

## Implementation Phases

### Phase 1: Foundation Layer ✅ COMPLETE
- [x] pyproject.toml, requirements.txt
- [x] .env.example, .gitignore
- [x] All YAML configs (models, training, retrieval, verification, evaluation, experiments)
- [x] Core Pydantic schemas (src/models/schemas.py)
- [x] Configuration loader (src/utils/config.py)
- [x] Structured logging (src/utils/logging.py)
- [x] All package __init__.py files

### Phase 2: Model Providers ✅ COMPLETE
- [x] Abstract base classes (src/models/base.py)
- [x] MedGemma provider (src/models/medgemma.py)
- [x] Qwen verifier provider (src/models/qwen_verifier.py)
- [x] Mock providers (src/models/mock_providers.py)
- [x] Model factory (src/models/factory.py)

### Phase 3: Retrieval Engine ✅ COMPLETE
- [x] Document loader (src/ingestion/document_loader.py)
- [x] Structure-aware chunker (src/preprocessing/chunker.py)
- [x] BM25 retriever (src/retrieval/bm25_retriever.py)
- [x] MedCPT dense retriever (src/retrieval/dense_retriever.py)
- [x] Hybrid RRF fusion (src/retrieval/fusion.py)
- [x] Cross-encoder reranker (src/retrieval/reranker.py)
- [x] Retrieval pipeline (src/retrieval/pipeline.py)
- [x] Retrieval evaluation metrics (src/retrieval/evaluation.py)

### Phase 4: Training Pipeline ✅ COMPLETE (code implemented, execution requires GPU)
- [x] SFT dataset preparation (src/training/sft/dataset.py)
- [x] RAFT dataset preparation (src/training/raft/dataset.py)
- [x] QLoRA configuration (src/training/qlora/config.py)
- [x] SFT trainer (src/training/sft/trainer.py)
- [x] RAFT trainer (src/training/raft/trainer.py)
- [x] Training utilities (src/training/utils.py)
- [x] SFT training script (scripts/train_sft.py) — dry-run validated
- [x] RAFT training script (scripts/train_raft.py) — dry-run validated

### Phase 5: Verification & Abstention ✅ COMPLETE
- [x] Claim extractor (src/verification/claim_extractor.py)
- [x] Citation validator (src/verification/citation_validator.py)
- [x] Entailment checker (src/verification/entailment.py)
- [x] Verification pipeline (src/verification/pipeline.py)
- [x] Decision engine with calibrated abstention (src/verification/decision.py)

### Phase 6: Pipeline Orchestration ✅ COMPLETE
- [x] End-to-end orchestrator (src/pipeline/orchestrator.py)
- [x] Routing framework with rule-based classifier (src/routing/router.py)
- [x] Factory function for experiment-driven orchestrator creation

### Phase 7: Evaluation Framework ✅ COMPLETE
- [x] Answer metrics (src/evaluation/answer_metrics.py)
- [x] Grounding metrics (src/evaluation/grounding_metrics.py)
- [x] Calibration metrics (src/evaluation/calibration.py)
- [x] Statistical tests (src/evaluation/statistical.py)
- [x] Performance metrics (src/evaluation/performance.py)
- [x] Evaluation runner (src/evaluation/runner.py)
- [x] Evaluation CLI script (scripts/evaluate.py)

### Phase 8: Dataset Engineering ✅ COMPLETE
- [x] Dataset preparation script (scripts/prepare_data.py) — executed, data generated
- [x] Dataset splits: train, validation, test (document-level grouping)
- [x] SFT and RAFT formatted datasets
- [x] Dataset report (data/processed/dataset_report.json)

### Phase 9: API & Frontend ✅ COMPLETE
- [x] FastAPI backend (backend/app.py, backend/api/routes.py)
- [x] API schemas (backend/schemas/requests.py, responses.py)
- [x] API services (backend/services/pipeline_service.py)
- [x] Streamlit frontend (frontend/app.py)
- [x] Pipeline CLI scripts (scripts/run_pipeline.py, scripts/run_baselines.py)

### Phase 10: Testing & Documentation ✅ COMPLETE
- [x] Unit tests: 33 tests (schemas, chunker, claim_extractor, citation_validator, decision, config)
- [x] Integration tests: 13 tests (mock_pipeline, api)
- [x] Component tests: 17 tests (models, retrieval)
- [x] **63 total tests, all passing** (2.95s)
- [x] README.md (444 lines)
- [x] Research documentation (10 documents)
- [x] Engineering report (research/engineering_report.md)
- [x] Docker support (Dockerfile, docker-compose.yml)
- [x] Jupyter notebook (notebooks/quickstart.ipynb)

## Execution Status

| Phase | Code | Tests | Execution |
|-------|------|-------|-----------|
| Foundation | ✅ | ✅ | ✅ |
| Model Providers | ✅ | ✅ | ✅ (mock) |
| Retrieval | ✅ | ✅ | ✅ (mock) |
| Training | ✅ | ✅ (dry-run) | ⏳ Requires GPU |
| Verification | ✅ | ✅ | ✅ (mock) |
| Orchestration | ✅ | ✅ | ✅ (mock) |
| Evaluation | ✅ | ✅ | ⏳ Requires trained models |
| Dataset | ✅ | ✅ | ✅ |
| API & Frontend | ✅ | ✅ | ✅ (mock) |
| Testing & Docs | ✅ | ✅ | ✅ |

## Technical Risks

1. **MedGemma gating**: Requires HF token acceptance; may block automated CI
2. **Qwen3.5-9B hybrid architecture**: Novel GatedDeltaNet may have limited PEFT support (but we're not fine-tuning it)
3. **Memory constraints**: Running both 4B + 9B models quantized requires ~12-16GB VRAM
4. **Python 3.13 compatibility**: Some ML packages may not fully support 3.13 yet
5. **RAFT training data quality**: Generating high-quality distractor passages requires careful engineering
6. **BitsAndBytes Windows**: Limited Windows support for bitsandbytes; Linux recommended for training

## Dependency Order

```
configs → schemas → utils → ingestion → preprocessing → retrieval → models → training → verification → routing → pipeline → evaluation → api → frontend
```
