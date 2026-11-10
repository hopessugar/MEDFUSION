# MedFusion-RAFT Limitations

## Model Limitations

### MedGemma Generator
- Medical knowledge limited to training data cutoff date
- May hallucinate medical facts even with evidence grounding
- Quantization (4-bit NF4) reduces model precision
- Performance on rare conditions or specialties is unvalidated
- Chat template sensitivity may affect output quality
- Gated model requires HuggingFace access approval

### Qwen Verifier
- Not a medical model — general-purpose instruction following
- May miss subtle medical inaccuracies or nuanced errors
- Verification quality is bounded by the verifier's medical understanding
- Shared pretraining data with MedGemma limits independence
- VLM architecture loaded as text-only may not be optimal

### MedCPT Retrieval
- Trained primarily on PubMed — biased toward biomedical literature
- May underperform on clinical guidelines, drug information, or patient-facing content
- Fixed maximum sequence lengths (64 query, 512 article)
- Encoding quality degrades on non-English text

## System Limitations

### Retrieval Dependence
- Answer quality is bounded by retrieval quality
- Missing relevant passages cannot be compensated by the generator
- BM25 requires exact term matching; misses paraphrased evidence
- Dense retrieval may retrieve topically related but factually irrelevant passages

### Verification Reliability
- Citation validation uses keyword overlap (heuristic, not semantic)
- NLI model trained on general NLI data, not medical entailment
- LLM verification is probabilistic, not deterministic
- Generator-verifier error correlation is unknown until measured
- Verification adds latency (second model inference)

### Abstention Calibration
- Platt scaling requires sufficient validation data
- Calibration may not generalize to out-of-distribution questions
- Threshold tuning is dataset-specific
- Binary abstention loses partial-confidence information

### Scalability
- Dual-model inference (4B + 9B) requires significant GPU memory
- Sequential model loading increases latency
- FAISS flat index does not scale to millions of documents
- No streaming / incremental indexing support

## Research Limitations

### Evaluation
- No expert clinical evaluation of answer quality
- Automated metrics are proxies for medical correctness
- LLM-as-a-judge scores require validation against human judgments
- Small dataset sizes limit statistical power
- Single hardware configuration limits generalizability

### Training
- SFT and RAFT data quality depends on source dataset quality
- Distractor passage selection strategy may be suboptimal
- No adversarial training or robustness testing
- Limited hyperparameter search due to compute constraints

### Safety
- No adversarial testing against prompt injection via documents
- No red-teaming for medical misinformation generation
- No evaluation on safety-critical edge cases (drug interactions, dosing)
- No multilingual or cross-cultural evaluation
- System may exhibit biases present in training data

## Scope Exclusions

This project explicitly does NOT:
- Provide clinical decision support
- Replace medical professional judgment
- Claim clinical validation or regulatory approval
- Handle medical images (text-only despite MedGemma's multimodal capability)
- Support real-time clinical workflows
- Guarantee factual accuracy of any generated answer
- Process identifiable patient data

