# MedFusion-RAFT Dataset Card

## Custom Medical Document QA Dataset

### Purpose
Primary evaluation dataset for the MedFusion-RAFT system. Designed to test evidence-grounded medical question answering with explicit answerability labels and citation requirements.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| question_id | string | Unique identifier |
| question | string | Medical question |
| reference_answer | string | Verified reference answer |
| gold_passage | string | Gold evidence passage text |
| gold_passage_id | string | Stable passage identifier |
| doc_id | string | Source document identifier |
| page | int | Page number in source |
| specialty | string | Medical specialty (e.g., cardiology, oncology) |
| difficulty | enum | easy, medium, hard |
| answerability | enum | answerable, partially_answerable, unanswerable |
| source_url | string | URL to source document |
| review_status | string | unreviewed, reviewed, verified |

### Answerability Categories

1. **Answerable**: The gold passage contains sufficient information to fully answer the question.
2. **Partially answerable**: The gold passage provides some relevant information but not a complete answer.
3. **Unanswerable**: The available evidence does not contain the information needed to answer the question.

### Split Strategy
- Split by source document (doc_id), NOT by random question splitting
- This prevents data leakage from related questions about the same document
- Minimum 3 questions per document for inclusion

### Quality Requirements
- No identifiable patient data
- No fabricated medical evidence
- Reference answers verified against source documents
- Duplicate detection using question similarity
- No overlap with standard benchmark test sets

## Public Benchmark Datasets

### PubMedQA
- **Source**: `pubmed_qa` on HuggingFace
- **License**: MIT
- **Use**: Biomedical yes/no/maybe QA from PubMed abstracts
- **Size**: ~1,000 expert-labeled, ~211,000 artificially generated

### MedQA (USMLE)
- **Source**: `bigbio/med_qa`
- **License**: MIT
- **Use**: Multiple-choice medical questions from USMLE
- **Size**: ~12,000 questions

### MedMCQA
- **Source**: `medmcqa`
- **License**: Apache 2.0
- **Use**: Multiple-choice from Indian medical entrance exams
- **Size**: ~194,000 questions

### MMLU Medical Subsets
- **Source**: `cais/mmlu` (anatomy, clinical_knowledge, college_medicine, medical_genetics, professional_medicine)
- **License**: MIT
- **Use**: Broad medical knowledge assessment

## Data Processing Pipeline

```
Raw sources → Document loading → Text extraction → 
Structure-aware chunking → Passage indexing → 
Dataset construction → Validation → Split → Export
```

## Licensing and Ethics

- All datasets used under their respective open-source licenses
- No identifiable patient data in any dataset
- No clinical validation claimed
- Generated answers are NOT medical advice
- Dataset provenance tracked and documented
- Terms of use verified before training

