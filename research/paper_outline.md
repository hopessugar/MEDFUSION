# MedFusion-RAFT: Paper Outline

## Title
MedFusion-RAFT: An Adaptive, Evidence-Grounded Medical Question Answering Framework with Retrieval-Augmented Fine-Tuning, Independent Verification and Selective Abstention

## Abstract
- Problem: LLMs generate plausible but potentially unsupported medical answers
- Approach: Multi-model pipeline combining fine-tuned medical generator, hybrid biomedical retrieval, independent verification, and calibrated abstention
- Key results: [TO BE FILLED AFTER EXPERIMENTS]
- Significance: Demonstrates measurable improvement in answer reliability through independent verification and selective abstention

## 1. Introduction
- Medical QA reliability challenge
- Limitations of single-model approaches
- Research question and contributions

## 2. Related Work
- Medical language models
- Retrieval-augmented generation for healthcare
- RAFT methodology
- Factual verification and claim checking
- Selective prediction / abstention

## 3. Method
### 3.1 System Architecture
- Multi-model pipeline overview
- Component responsibilities and interfaces

### 3.2 Biomedical Hybrid Retrieval
- MedCPT dense retrieval
- BM25 sparse retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking

### 3.3 Evidence-Grounded Generation
- MedGemma architecture
- Structured output format
- Citation requirements

### 3.4 Parameter-Efficient Fine-Tuning
- QLoRA configuration
- Stage 1: Supervised Fine-Tuning
- Stage 2: Retrieval-Augmented Fine-Tuning
- Training data construction (relevant passages + distractors)

### 3.5 Independent Verification
- Claim decomposition
- Deterministic citation validation
- NLI-based entailment checking
- LLM-based verification (Qwen3.5)
- Multi-signal integration

### 3.6 Selective Abstention
- Composite confidence scoring
- Platt scaling calibration
- Accept/Revise/Abstain decision policy

### 3.7 Adaptive Routing
- Question classification
- Resource allocation based on complexity

## 4. Experimental Setup
### 4.1 Datasets
- Custom medical document QA dataset
- PubMedQA, MedQA benchmarks
- Split strategy (document-level)

### 4.2 Evaluation Metrics
- Answer quality (accuracy, F1, EM)
- Retrieval quality (Recall@K, MRR, nDCG@K)
- Grounding quality (claim support, citation precision)
- Abstention quality (selective risk, coverage)
- Calibration (ECE, Brier score)
- Efficiency (latency, memory)

### 4.3 Experiment Matrix
- E1-E9 ablation study
- Component-level ablations

### 4.4 Implementation Details
- Hardware specifications
- Hyperparameters
- Training details

## 5. Results
### 5.1 Main Results
- [TABLE: E1-E9 comparison on all metrics]

### 5.2 Retrieval Analysis
- Impact of hybrid vs. single retrieval
- Retrieval quality vs. answer quality correlation

### 5.3 Fine-Tuning Analysis
- SFT vs. RAFT impact
- Distractor robustness improvement

### 5.4 Verification Analysis
- Generator-verifier error correlation
- False acceptance/rejection rates
- Value of deterministic vs. LLM verification

### 5.5 Abstention Analysis
- Coverage-risk curves
- Calibration quality
- Abstention precision/recall

### 5.6 Ablation Study
- [TABLE: Component-level contributions]

### 5.7 Efficiency Analysis
- Latency breakdown by component
- Memory usage
- Cost per query

## 6. Discussion
- Key findings
- When does verification help most?
- Accuracy-coverage trade-offs
- Limitations of the approach
- Clinical applicability considerations

## 7. Conclusion
- Summary of contributions
- Practical implications
- Future work

## References
[See literature_review.md for full bibliography]

## Status: OUTLINE ONLY
This paper has not been written. Results sections will be populated only after experiments are executed and results are available.

