# MedFusion-RAFT Experiment Protocol

## Experimental Design

### Research Question
Can a coordinated medical generator, biomedical retrieval system and independent verifier produce more reliable evidence-grounded medical answers than an individual language model, while maintaining a reasonable accuracy, latency and computational-cost trade-off?

### Independent Variables
1. Fine-tuning method (none, SFT, RAFT, SFT+RAFT)
2. Retrieval strategy (none, BM25, hybrid)
3. Verification (none, independent Qwen verification)
4. Abstention policy (none, calibrated selective abstention)
5. Routing strategy (none, rule-based adaptive)

### Dependent Variables
1. Answer accuracy / F1
2. Retrieval quality (Recall@5, MRR, nDCG@5)
3. Grounding quality (claim support rate, citation precision)
4. Abstention quality (selective risk, coverage)
5. Calibration (ECE, Brier score)
6. Latency (median, P95)
7. GPU memory usage

## Experiment Matrix

| ID | Configuration | Fine-tuning | Retrieval | Verification | Abstention |
|----|---------------|-------------|-----------|--------------|------------|
| E1 | Base MedGemma | None | None | No | No |
| E2 | MedGemma + BM25 | None | BM25 | No | No |
| E3 | MedGemma + Hybrid RAG | None | Hybrid | No | No |
| E4 | MedGemma + SFT | SFT | None | No | No |
| E5 | MedGemma + RAFT | RAFT | Hybrid | No | No |
| E6 | MedGemma + SFT + RAFT + RAG | SFT→RAFT | Hybrid | No | No |
| E7 | E6 + Verification | SFT→RAFT | Hybrid | Yes | No |
| E8 | E7 + Abstention | SFT→RAFT | Hybrid | Yes | Yes |
| E9 | Complete System | SFT→RAFT | Hybrid | Yes | Yes + Routing |

### Ablation Studies
- E6 minus BM25 (dense-only retrieval)
- E6 minus dense (BM25-only retrieval)
- E6 minus reranking
- E7 minus NLI entailment (LLM-only verification)
- E7 minus LLM verification (deterministic-only verification)
- E8 minus calibration (raw threshold abstention)

## Evaluation Protocol

### Dataset Splits
- **Training**: 70% of examples (split by source document)
- **Validation**: 15% of examples (for hyperparameter tuning, calibration)
- **Test**: 15% of examples (held-out, never used for tuning)

### Statistical Rigor
- 95% bootstrap confidence intervals (1000 samples)
- Paired bootstrap tests for experiment comparisons
- McNemar's test for accuracy comparisons
- Bonferroni correction for multiple comparisons
- Report effect sizes alongside p-values

### Seeds
- Primary seed: 42
- Additional seeds for variance estimation: 123, 456 (when compute allows)

### Reporting Requirements
- All results must come from actual experiment runs
- NOT_RUN status for experiments not yet executed
- No fabricated numbers, trends, or improvements
- Report both mean and confidence intervals
- Clearly distinguish training/validation/test performance

## Hardware Requirements

### Minimum (development + mock testing)
- CPU: 4 cores
- RAM: 16 GB
- Storage: 20 GB
- GPU: None required

### Recommended (full pipeline execution)
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 100 GB
- GPU: 1x NVIDIA GPU with ≥24 GB VRAM (e.g., RTX 3090, A5000)

### Full training + dual model inference
- CPU: 16 cores
- RAM: 64 GB
- Storage: 200 GB
- GPU: 1x NVIDIA GPU with ≥40 GB VRAM (e.g., A100-40GB, A6000)

## Reproducibility Checklist

- [ ] Fixed random seeds across all components
- [ ] Pinned dependency versions
- [ ] Recorded base model revisions
- [ ] Saved training configurations as JSON
- [ ] Saved LoRA adapters (not merged weights)
- [ ] Recorded hardware info with results
- [ ] Documented dataset preparation steps
- [ ] Version-controlled experiment configs
- [ ] Saved tokenizer with adapters
- [ ] Recorded generation parameters

