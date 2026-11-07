# MedFusion-RAFT: Research Validity & Scientific Methodology Audit

**Audit Date**: 2026-11-08  
**Auditor**: LLM Research Scientist & Medical AI Specialist  
**Evaluation Target**: Methodological Soundness, RAFT Fidelity, Metric Validity, and Reproducibility  

---

## 1. Research Problem & Core Hypothesis

**Research Title**:  
*MedFusion-RAFT: An Adaptive, Evidence-Grounded Medical Question Answering Framework with Retrieval-Augmented Fine-Tuning, Independent Verification and Selective Abstention.*

**Core Hypotheses**:
1. Fine-tuning a medical LLM with RAFT (training on mixtures of gold evidence and irrelevant distractor passages) significantly increases the model's robustness against retrieval noise and hallucination compared to standard SFT.
2. Employing an independent, distinct model family (Alibaba Qwen3.5-9B) to decompose and verify atomic claims provides factual grounding checks that reduce uncorrelated errors.
3. Combining composite uncertainty estimation with calibrated selective abstention (Platt scaling) produces a strictly monotonic risk-coverage curve, allowing the system to withhold answers on ungrounded or ambiguous queries.

---

## 2. RAFT Methodology Fidelity Audit

We audited `src/training/raft/dataset.py` and `src/training/raft/trainer.py` to determine whether the RAFT implementation is genuine or merely standard RAG prompt-stuffing with a RAFT label.

### Audit Findings on RAFT Implementation:
1. **Distractor Passage Mixing**:
   - `RAFTDatasetBuilder` takes a pool of relevant and distractor passages per question.
   - Configurable `distractor_ratio` (default `0.5`) determines the probability that a given training instance contains distractor passages alongside or in place of gold passages.
   - Verified in `src/training/raft/dataset.py` lines 90-145: Passages are shuffled and indexed with bracketed citation markers `[P1]`, `[P2]`, `[P3]`, training the generator to extract and cite only genuine evidence while ignoring distractors.
2. **Handling Unanswerable Questions**:
   - The builder explicitly formats unanswerable examples where all retrieved passages are distractors, prompting the model to generate:
     `INSUFFICIENT_EVIDENCE: The available documentation does not contain sufficient clinical information to answer this question.`
   - This directly follows the Zhang et al. (2024) RAFT specification for unanswerable instances.
3. **Two-Stage SFT -> RAFT Training Pipeline**:
   - `MedFusionRAFTTrainer` in `src/training/raft/trainer.py` accepts `--sft-adapter` from Stage 1. It loads the base MedGemma model, attaches the trained SFT LoRA adapter, merges or stacks the weights, and trains a dedicated RAFT adapter.
   - **Verdict**: **GENUINE RAFT IMPLEMENTATION VERIFIED**.

---

## 3. Evaluation Methodology & Metrics Rigor

The evaluation framework in `src/evaluation/` implements a rigorous multi-dimensional assessment:

### A. Generation Quality Metrics (`answer_metrics.py`)
- **Exact Match (EM)**: Normalized token matching after lowercase/punctuation removal.
- **Token F1**: Precision and recall over word tokens.
- **ROUGE-L**: Longest common subsequence matching for clinical explanation completeness.
- **BLEU-4**: Smoothed 4-gram precision matching against reference answers.

### B. Grounding & Verification Metrics (`grounding_metrics.py`)
- **Claim Support Rate**: Fraction of extracted atomic claims classified as SUPPORTED by evidence.
- **Contradiction Rate**: Fraction of claims classified as CONTRADICTED by evidence.
- **Citation Precision**: Fraction of citations that point to an existing, keyword-relevant retrieved passage.
- **Citation Recall**: Fraction of required evidence passages successfully cited.

### C. Calibration & Selective Prediction Metrics (`calibration.py`)
- **Expected Calibration Error (ECE)**: Equal-width binning measuring the difference between predicted confidence and observed empirical accuracy.
- **Brier Score**: Mean squared error between predicted probabilities and binary correctness labels.
- **Selective Risk & Coverage**: Computes accuracy at varying coverage thresholds ($C \in [0.1, 1.0]$) to confirm monotonicity.

### D. Statistical Testing Rigor (`statistical.py`)
- **Paired Non-Parametric Bootstrap Test**: 1,000 bootstrap iterations to compute $p$-values for performance differences between experimental tiers without assuming normality.
- **McNemar's Paired Test**: Exact McNemar test for paired binary accuracy comparisons between baseline (E1) and proposed (E9) systems.
- **Bonferroni Correction**: Controls family-wise error rate across multiple ablation hypothesis tests.

---

## 4. Model Independence & Verification Analysis

The project specifies `google/medgemma-1.5-4b-it` as the generator and `Qwen/Qwen3.5-9B` as the verifier.

### Assessment of Model Independence:
- **Architectural Diversity**: MedGemma is a Dense Transformer based on Gemma 3 (Google). Qwen3.5 is a hybrid GatedDeltaNet + Attention architecture (Alibaba).
- **Training Corpus Diversity**: The pretraining corpora are distinct (Google's internal datasets vs. Alibaba's open-web/multilingual datasets), significantly reducing the risk of shared pretraining hallucinations.
- **Decoupled Responsibilities**:
  - MedGemma is fine-tuned to synthesize answers and generate citation markers.
  - Qwen is kept **frozen** and prompted strictly as an adversarial reviewer to decompose claims, compare each proposition against passage text, and assess entailment.
- **Deterministic Safeguards**: The pipeline does NOT rely solely on Qwen's judgment. Deterministic citation checks and DeBERTa-v3 NLI cross-validation run in parallel. Even if Qwen hallucinated agreement, a contradicted claim detected by NLI or a missing citation is caught deterministically.

---

## 5. Reproducibility & Scientific Honesty

1. **Random Seed Management**:
   - `src/training/utils.py::set_seed()` sets identical seeds across Python `random`, NumPy, and PyTorch (CPU and CUDA deterministic flags enabled).
2. **Data Leakage Prevention**:
   - `scripts/prepare_data.py` splits datasets strictly by `doc_id`. Multiple questions derived from the same source clinical document are never split across train and test sets.
3. **Execution Status Transparency**:
   - In `research/paper_outline.md`, `research/error_analysis.md`, and `scripts/run_baselines.py`, sections requiring GPU-trained model execution are explicitly labeled:
     `Status: NOT YET EXECUTED — Results will be populated after GPU experiments are run.`
   - No benchmark numbers or experimental scores have been fabricated. Mock runs are explicitly designated with `"mode": "mock"`.
