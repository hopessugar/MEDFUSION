# MedFusion-RAFT

**An Adaptive, Evidence-Grounded Medical Question Answering Framework with Retrieval-Augmented Fine-Tuning, Independent Verification and Selective Abstention**



---

## Overview

MedFusion-RAFT is a modular, research-grade medical question-answering system that combines:

1. **Medical-domain model adaptation** — QLoRA fine-tuning of MedGemma 1.5-4b-it
2. **Hybrid biomedical retrieval** — MedCPT + BM25 with Reciprocal Rank Fusion
3. **Retrieval-Augmented Fine-Tuning (RAFT)** — Training with relevant and distractor passages
4. **Independent verification** — Qwen3.5-9B from a different model family
5. **Evidence-based generation** — Claims linked to cited passage IDs
6. **Calibrated selective abstention** — Platt-scaled confidence with accept/revise/abstain decisions
7. **Rigorous evaluation** — Statistical tests, ablation studies, and component-level metrics

### Research Question

> Can a coordinated medical generator, biomedical retrieval system and independent verifier produce more reliable evidence-grounded medical answers than an individual language model, while maintaining a reasonable accuracy, latency and computational-cost trade-off?

---

## Architecture

```
USER QUESTION
    ↓
INPUT VALIDATION → QUESTION CLASSIFICATION
    ↓
HYBRID BIOMEDICAL RETRIEVAL
  ├── BM25 Sparse Retrieval
  ├── MedCPT Dense Retrieval
  ├── Reciprocal Rank Fusion
  └── Cross-Encoder Reranking
    ↓
MEDGEMMA 1.5-4b-it GENERATION
  └── Structured output: answer + citations + answerability
    ↓
VERIFICATION PIPELINE
  ├── Atomic Claim Extraction
  ├── Deterministic Citation Validation
  ├── NLI Entailment Check (DeBERTa-v3)
  └── Qwen3.5-9B Independent Verification
    ↓
DECISION POLICY
  ├── ACCEPT → Return answer
  ├── REVISE → Regenerate with feedback (limited)
  └── ABSTAIN → Return abstention notice
    ↓
STRUCTURED RESPONSE
```

### Models

| Role | Model | Parameters | Family | Status |
|------|-------|-----------|--------|--------|
| Generator | `google/medgemma-1.5-4b-it` | 4B | Gemma 3 | Fine-tuned (QLoRA) |
| Verifier | `Qwen/Qwen3.5-9B` | 9B | Qwen 3.5 | Frozen pretrained |
| Dense Retrieval | `ncbi/MedCPT-*-Encoder` | ~110M each | BERT-based | Frozen pretrained |
| Entailment | `cross-encoder/nli-deberta-v3-base` | ~86M | DeBERTa-v3 | Frozen pretrained |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- (Optional) NVIDIA GPU with ≥ 16 GB VRAM for model inference
- (Optional) ≥ 24 GB VRAM for training
- HuggingFace account with MedGemma access approved

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/medfusion-raft.git
cd medfusion-raft

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
# Or: pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### CPU-Only Testing (Mock Models)

```bash
# Run mock pipeline (no GPU or model downloads needed)
python scripts/run_pipeline.py --mock --question "What are the first-line treatments for hypertension?"

# Run tests
pytest tests/ -v

# Start API with mock models
python -m backend.app --mock
```

### Full Pipeline (GPU Required)

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run pipeline with real models
python scripts/run_pipeline.py --question "What are the first-line treatments for hypertension?"

# Start API server
uvicorn backend.app:create_app --host 0.0.0.0 --port 8000

# Start Streamlit UI
streamlit run frontend/app.py
```

---

## Fine-Tuning

### Dataset Preparation

```bash
# Prepare SFT and RAFT training datasets
python scripts/prepare_data.py \
    --output-dir data/processed \
    --include-pubmedqa \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py \
    --model-config configs/models.yaml \
    --training-config configs/training.yaml \
    --data-dir data/processed \
    --output-dir experiments/checkpoints/sft \
    --seed 42
```

### Stage 2: Retrieval-Augmented Fine-Tuning (RAFT)

```bash
python scripts/train_raft.py \
    --model-config configs/models.yaml \
    --training-config configs/training.yaml \
    --data-dir data/processed \
    --sft-adapter experiments/checkpoints/sft/best \
    --output-dir experiments/checkpoints/raft \
    --seed 42
```

### QLoRA Configuration

| Parameter | SFT | RAFT |
|-----------|-----|------|
| Quantization | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| Learning rate | 2e-5 | 1e-5 |
| Batch size | 1 | 1 |
| Gradient accumulation | 8 | 8 |
| Epochs | 2 | 2 |
| Max sequence length | 2048 | 2048 |
| Optimizer | Paged AdamW 8-bit | Paged AdamW 8-bit |

---

## Evaluation

### Run Evaluation

```bash
# Single experiment
python scripts/evaluate.py \
    --experiment-id E1_base \
    --config-dir configs \
    --data-path data/processed/test.jsonl \
    --output-dir experiments/results

# Full experiment matrix
python scripts/evaluate.py \
    --run-all \
    --config-dir configs \
    --data-path data/processed/test.jsonl \
    --output-dir experiments/results
```

### Experiment Matrix

| ID | Configuration | Purpose |
|----|---------------|---------|
| E1 | Base MedGemma | Baseline without any augmentation |
| E2 | + BM25 RAG | Effect of sparse retrieval |
| E3 | + Hybrid RAG | Effect of dense + sparse retrieval |
| E4 | + SFT | Effect of supervised fine-tuning |
| E5 | + RAFT | Effect of RAFT training |
| E6 | + SFT + RAFT + Hybrid | Combined fine-tuning + retrieval |
| E7 | + Verification | Effect of independent verification |
| E8 | + Abstention | Effect of selective abstention |
| E9 | + Adaptive Routing | Complete system |

### Metrics

- **Answer quality**: Accuracy, F1, Exact Match, ROUGE-L
- **Retrieval**: Recall@K, Precision@K, MRR, nDCG@K
- **Grounding**: Claim support rate, citation precision/recall, unsupported claim rate
- **Abstention**: Selective risk, coverage, abstention precision/recall
- **Calibration**: ECE, Brier score
- **Performance**: Median/P95 latency, peak GPU memory, tokens/second

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ask` | Ask a single question |
| POST | `/api/v1/batch` | Batch question processing |
| POST | `/api/v1/evaluate` | Run evaluation on dataset |
| GET | `/api/v1/health` | System health check |
| GET | `/api/v1/config` | Current configuration |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the contraindications for metformin?"}'
```

### Example Response

```json
{
    "answer_text": "Metformin is contraindicated in patients with...",
    "citations": ["P1", "P3"],
    "answerability": "answerable",
    "confidence": 0.82,
    "claims": [
        {
            "claim_text": "Metformin is contraindicated in renal impairment",
            "support_status": "supported",
            "evidence_passage_id": "P1"
        }
    ],
    "verification_status": "verified",
    "decision": "accept",
    "processing_time_ms": 3450.2
}
```

---

## Docker

```bash
# Build
docker build -t medfusion-raft .

# Run API
docker compose up api

# Run with GPU
docker compose --profile gpu up api

# Run with UI
docker compose up api frontend
```

---

## Project Structure

```
medfusion-raft/
├── configs/                    # YAML configuration files
│   ├── models.yaml             # Model IDs, quantization, generation
│   ├── training.yaml           # QLoRA, SFT, RAFT hyperparameters
│   ├── retrieval.yaml          # Chunking, BM25, dense, fusion
│   ├── verification.yaml       # Claims, citations, entailment, abstention
│   ├── evaluation.yaml         # Metrics, statistical tests
│   └── experiments.yaml        # Experiment matrix (E1-E9)
├── src/
│   ├── models/                 # Model providers and schemas
│   │   ├── schemas.py          # All Pydantic data models
│   │   ├── base.py             # Abstract model interfaces
│   │   ├── medgemma.py         # MedGemma generator provider
│   │   ├── qwen_verifier.py    # Qwen verifier provider
│   │   ├── mock_providers.py   # Mock providers for testing
│   │   └── factory.py          # Model factory
│   ├── ingestion/              # Document loading
│   ├── preprocessing/          # Text chunking
│   ├── retrieval/              # Retrieval engine
│   │   ├── bm25_retriever.py   # BM25 sparse retrieval
│   │   ├── dense_retriever.py  # MedCPT dense retrieval
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── pipeline.py         # Retrieval orchestration
│   │   └── evaluation.py       # Retrieval metrics
│   ├── training/               # Fine-tuning pipeline
│   │   ├── qlora/config.py     # QLoRA configuration
│   │   ├── sft/                # SFT dataset + trainer
│   │   └── raft/               # RAFT dataset + trainer
│   ├── verification/           # Verification system
│   │   ├── claim_extractor.py  # Atomic claim extraction
│   │   ├── citation_validator.py # Deterministic citation checks
│   │   ├── entailment.py       # NLI entailment checking
│   │   ├── pipeline.py         # Verification orchestration
│   │   └── decision.py         # Accept/Revise/Abstain policy
│   ├── routing/                # Question routing
│   ├── evaluation/             # Evaluation framework
│   └── pipeline/               # End-to-end orchestrator
├── backend/                    # FastAPI backend
├── frontend/                   # Streamlit UI
├── scripts/                    # CLI scripts
├── tests/                      # Unit + integration tests
├── research/                   # Research documentation
├── experiments/                # Experiment outputs
└── data/                       # Datasets
```

---

## Resource Requirements

| Configuration | GPU VRAM | RAM | Storage |
|--------------|----------|-----|---------|
| Mock testing (CPU) | 0 GB | 8 GB | 5 GB |
| Inference only | 16 GB | 16 GB | 30 GB |
| Single model training | 24 GB | 32 GB | 50 GB |
| Full pipeline (dual model) | 40+ GB | 64 GB | 100 GB |

---

## Reproducibility

1. All random seeds are configurable and default to 42
2. Dependency versions are pinned in `pyproject.toml`
3. Base model revisions are recorded in config
4. LoRA adapters are saved separately (base model not modified)
5. Training configs are saved as JSON with each checkpoint
6. Hardware info is recorded with experiment results
7. Dataset splits use document-level grouping (not random)
8. Statistical tests use bootstrap confidence intervals

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Configuration system | ✅ Implemented | YAML configs, Pydantic validation |
| Data schemas | ✅ Implemented | 20+ Pydantic models |
| Document ingestion | ✅ Implemented | TXT, JSON, JSONL, MD, CSV |
| Structure-aware chunking | ✅ Implemented | Section/sentence preservation |
| BM25 retrieval | ✅ Implemented | With fallback implementation |
| MedCPT dense retrieval | ✅ Implemented | FAISS index + numpy fallback |
| Hybrid RRF fusion | ✅ Implemented | Configurable weights |
| Cross-encoder reranking | ✅ Implemented | MedCPT cross-encoder |
| MedGemma provider | ✅ Implemented | QLoRA quantization support |
| Qwen verifier provider | ✅ Implemented | Text-only CausalLM loading |
| Mock providers | ✅ Implemented | Multiple modes |
| Model factory | ✅ Implemented | Extensible registration |
| Claim extraction | ✅ Implemented | Sentence splitting, citation parsing |
| Citation validation | ✅ Implemented | Keyword relevance checking |
| NLI entailment | ✅ Implemented | DeBERTa-v3 + mock fallback |
| Verification pipeline | ✅ Implemented | Multi-stage verification |
| Decision engine | ✅ Implemented | Calibrated abstention |
| QLoRA config | ✅ Implemented | BitsAndBytes + PEFT |
| SFT dataset builder | ✅ Implemented | Chat template formatting |
| RAFT dataset builder | ✅ Implemented | Distractor passage mixing |
| SFT trainer | ✅ Implemented | TRL-based with MLflow |
| RAFT trainer | ✅ Implemented | Sequential from SFT |
| Pipeline orchestrator | ✅ Implemented | Full end-to-end flow |
| Question router | ✅ Implemented | Rule-based classification |
| Evaluation metrics | ✅ Implemented | Answer, retrieval, grounding |
| Statistical tests | ✅ Implemented | Bootstrap CIs, McNemar |
| Calibration metrics | ✅ Implemented | ECE, Brier, reliability |
| FastAPI backend | ✅ Implemented | REST API with health check |
| Streamlit frontend | ✅ Implemented | Research interface |
| Unit tests | ✅ Implemented | Schemas, chunking, verification |
| Integration tests | ✅ Implemented | Mock pipeline, API |
| Docker support | ✅ Implemented | Multi-stage build |
| Training execution | ⏳ Not executed | Requires GPU + model access |
| Full evaluation | ⏳ Not executed | Requires trained models |
| Experiment results | ⏳ Not executed | No fabricated results |

---

## Known Limitations

See [research/limitations.md](research/limitations.md) for comprehensive details.

Key limitations:
- Medical knowledge bounded by training data
- Verification is probabilistic, not guaranteed
- Calibration is dataset-specific
- No clinical validation performed
- No adversarial robustness testing

---

## Citation

```bibtex
@software{medfusion_raft_2024,
    title={MedFusion-RAFT: An Adaptive, Evidence-Grounded Medical Question
           Answering Framework with Retrieval-Augmented Fine-Tuning,
           Independent Verification and Selective Abstention},
    year={2024},
    url={https://github.com/your-org/medfusion-raft}
}
```

---

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

Individual model and dataset licenses:
- MedGemma: Gemma Terms of Use (Google)
- Qwen3.5: Apache 2.0 (Alibaba)
- MedCPT: MIT (NCBI/NLM)
- DeBERTa-v3 NLI: MIT
- PubMedQA: MIT

