# MedFusion-RAFT Literature Review

## Retrieval-Augmented Generation (RAG) for Medical QA

### Foundational Work
- **Lewis et al. (2020)** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Introduced RAG combining pretrained sequence-to-sequence models with a dense retriever. Foundation for our retrieval-augmented approach. [NeurIPS 2020]
- **Borgeaud et al. (2022)** — *Improving language models by retrieving from trillions of tokens* (RETRO). Showed chunk-level retrieval can improve factual accuracy at scale. [ICML 2022]

### Medical RAG
- **Xiong et al. (2024)** — *Benchmarking Retrieval-Augmented Generation for Medicine*. Comprehensive evaluation of RAG for medical QA including PubMedQA, MedQA. Found that retrieval quality is the primary bottleneck. [arXiv:2402.13178]
- **Zakka et al. (2024)** — *Almanac: Retrieval-Augmented Language Models for Clinical Medicine*. Clinical RAG system with medical guideline retrieval. [Nature Medicine]

### Retrieval-Augmented Fine-Tuning (RAFT)
- **Zhang et al. (2024)** — *RAFT: Adapting Language Model to Domain Specific RAG*. Proposed training with a mix of relevant and distractor documents to improve retrieval robustness. Core methodology we adapt for medical domain. [arXiv:2403.10131]

**Our contribution**: We adapt RAFT specifically for medical QA with answerability classification and multi-evidence reasoning, adding independent verification that the original RAFT paper does not include.

## Biomedical Information Retrieval

### Dense Retrieval
- **Jin et al. (2023)** — *MedCPT: Contrastive Pre-trained Transformers for Zero-shot Biomedical Information Retrieval*. Purpose-built biomedical retrieval model trained on PubMed click-through data. We use this as our primary dense retriever. [Bioinformatics]
- **Karpukhin et al. (2020)** — *Dense Passage Retrieval for Open-Domain Question Answering* (DPR). Foundation for our dual-encoder retrieval approach. [EMNLP 2020]

### Hybrid Retrieval
- **Cormack et al. (2009)** — *Reciprocal Rank Fusion*. We use RRF to combine BM25 and MedCPT retrieval results. [SIGIR 2009]

**Our contribution**: Engineering integration of MedCPT with BM25 via RRF, with medical cross-encoder reranking. These are established methods; our contribution is the integrated pipeline.

## Medical Language Models

### Medical Foundation Models
- **Saab et al. (2024)** — *Capabilities of Frontier Clinical Models in Medicine*. Analyzes foundational clinical domain adaptation. MedGemma is the open clinical foundation model we fine-tune. [Nature Medicine]
- **Singhal et al. (2023)** — *Towards Expert-Level Medical Question Answering with Large Language Models* (Med-PaLM 2). Set medical QA benchmarks. [Nature Medicine]

### Parameter-Efficient Fine-Tuning
- **Hu et al. (2021)** — *LoRA: Low-Rank Adaptation of Large Language Models*. Foundation for our QLoRA approach. [ICLR 2022]
- **Dettmers et al. (2023)** — *QLoRA: Efficient Finetuning of Quantized Language Models*. 4-bit quantization with LoRA. Core fine-tuning method we use. [NeurIPS 2023]

## Answer Verification and Factuality

### Claim Decomposition
- **Min et al. (2023)** — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. Atomic claim decomposition for factuality evaluation. Influences our claim extraction approach. [EMNLP 2023]

### NLI for Verification
- **Honovich et al. (2022)** — *TRUE: Re-evaluating Factual Consistency Evaluation*. NLI-based factual consistency checking. Our entailment checker uses a similar approach. [NAACL 2022]
- **He et al. (2023)** — *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training*. We use DeBERTa-v3 for NLI-based claim verification. [ICLR 2023]

### Multi-Model Verification
- **Cohen et al. (2023)** — *LM vs LM: Detecting Factual Errors via Cross Examination*. Using a different LLM to verify outputs. Influences our independent Qwen verification approach. [EMNLP 2023]

**Our contribution**: Combining deterministic citation validation, NLI entailment checking, and independent LLM verification is our engineering integration. Using models from different families (Gemma generator, Qwen verifier) to reduce correlated errors is a design choice we evaluate experimentally.

## Selective Prediction / Abstention

- **Geifman & El-Yaniv (2017)** — *Selective Classification for Deep Neural Networks*. Foundational framework for selective prediction that we adapt. [NeurIPS 2017]
- **Kamath et al. (2020)** — *Selective Question Answering under Domain Shift*. QA abstention under distribution shift. [ACL 2020]
- **Varshney et al. (2022)** — *Investigating Selective Prediction Approaches Across Several Tasks in IID, OOD, and Adversarial Settings*. Coverage-risk trade-off analysis. [ACL Findings 2022]

**Our contribution**: Calibrated abstention using multi-signal composite scoring (verification, citation quality, evidence availability, generator confidence) with Platt scaling calibration from validation data.

## Distinction of Contributions

| Category | Description |
|----------|-------------|
| **Existing methods** | RAG, RAFT, QLoRA, RRF, NLI verification, MedCPT, BM25 |
| **Our engineering integration** | Combining MedGemma generator + MedCPT hybrid retrieval + Qwen verification + NLI entailment + calibrated abstention into a single modular pipeline |
| **Our proposed methodological changes** | RAFT adaptation for medical domain with answerability labels; multi-signal abstention scoring; deterministic+LLM verification combination |
| **To be demonstrated experimentally** | Whether the integrated system outperforms individual components; the value of independent verification; the accuracy-coverage trade-off of calibrated abstention |

