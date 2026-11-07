# MedFusion-RAFT Model Card

## Generator: MedGemma 1.5-4b-it

### Model Details
- **Model ID**: `google/medgemma-1.5-4b-it`
- **Architecture**: Gemma 3 base, 4B parameters, instruction-tuned
- **Developer**: Google Health AI Developer Foundations
- **License**: Gemma Terms of Use (restricted)
- **Access**: Gated on HuggingFace (requires acceptance of terms)
- **Modality**: Multimodal (text + vision), used here for text-only

### Intended Use in MedFusion-RAFT
- Primary medical answer generator
- Fine-tuned with QLoRA for evidence-grounded answering
- NOT used for clinical decision-making

### Fine-tuning
- Method: QLoRA (4-bit NF4 quantization + LoRA adapters)
- Stage 1: Supervised Fine-Tuning (SFT) on medical QA pairs
- Stage 2: Retrieval-Augmented Fine-Tuning (RAFT) with distractor passages
- Adapters saved separately, base model not modified

### Limitations
- Medical knowledge limited to training data cutoff
- May hallucinate medical facts despite fine-tuning
- Requires independent verification of generated answers
- Not validated for clinical use

---

## Verifier: Qwen3.5-9B

### Model Details
- **Model ID**: `Qwen/Qwen3.5-9B`
- **Architecture**: Hybrid GatedDeltaNet + Attention, 9B parameters
- **Developer**: Alibaba Qwen Team
- **License**: Apache 2.0
- **Access**: Public on HuggingFace
- **Loading**: Text-only via `AutoModelForCausalLM` (strips vision encoder)
- **Chat Template**: ChatML (`<|im_start|>`)

### Intended Use in MedFusion-RAFT
- Independent answer verifier (different model family from generator)
- Decomposes answers into atomic claims
- Checks each claim against evidence passages
- Identifies unsupported or contradicted statements
- NOT fine-tuned (frozen pretrained weights)

### Independence Rationale
Using a different model family (Alibaba Qwen vs Google Gemma) helps reduce correlated errors. However, shared pretraining data may limit true independence — this is measured by computing generator-verifier error correlation.

### Limitations
- LLM verification is not infallible
- Complemented by deterministic citation and NLI checks
- May miss subtle medical inaccuracies
- Verification quality depends on evidence quality

---

## Retrieval: MedCPT

### Model Details
- **Query Encoder**: `ncbi/MedCPT-Query-Encoder` (BERT-based)
- **Article Encoder**: `ncbi/MedCPT-Article-Encoder` (BERT-based)
- **Cross-Encoder**: `ncbi/MedCPT-Cross-Encoder` (BERT-based, for reranking)
- **Developer**: NCBI/NLM
- **License**: MIT
- **Embedding Dim**: 768

### Intended Use in MedFusion-RAFT
- Dense biomedical passage retrieval
- Query-passage relevance scoring
- Cross-encoder reranking of retrieval candidates
- Combined with BM25 via Reciprocal Rank Fusion

### Limitations
- Trained primarily on PubMed data
- May underperform on non-biomedical medical content
- Maximum sequence lengths (64 for queries, 512 for passages)

---

## Entailment: DeBERTa-v3 NLI

### Model Details
- **Model ID**: `cross-encoder/nli-deberta-v3-base`
- **Architecture**: DeBERTa-v3, cross-encoder for NLI
- **Developer**: Sentence Transformers project
- **License**: MIT
- **Labels**: [contradiction, entailment, neutral]

### Intended Use in MedFusion-RAFT
- Deterministic entailment checking between claims and evidence
- Complements LLM-based verification
- Provides calibrated support scores

---

## Safety and Ethical Considerations

> [!CAUTION]
> MedFusion-RAFT is an experimental research prototype. It is NOT a clinically validated medical decision-support system. Generated answers should NOT be used for individual diagnosis or treatment decisions.

- No identifiable patient data used in training or evaluation
- No clinical validation performed
- Answers include explicit uncertainty indicators
- System can abstain when evidence is insufficient
- All source materials are cited with provenance
- Model limitations are clearly documented

