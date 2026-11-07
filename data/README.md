# MedFusion-RAFT Data Directory Specification

This directory holds clinical guideline corpora, raw source literature, processed passage chunks, retrieval indices, and formatted datasets for Supervised Fine-Tuning (SFT), Retrieval-Augmented Fine-Tuning (RAFT), and factual evaluation.

---

## Directory Organization

```text
data/
├── README.md               # Data directory layout and schema documentation
├── raw/                    # Original unformatted medical source files
│   ├── guidelines/         # Clinical practice guidelines (PDF, HTML, TXT)
│   ├── pubmed/             # Open-access PubMed Central articles and abstracts
│   └── benchmarks/         # Raw source benchmarks (MedQA, PubMedQA, BioASQ)
├── processed/              # Normalized, chunked, and split datasets (JSONL format)
│   ├── chunks/             # Structure-aware document passage chunks
│   ├── sft/                # SFT train/validation/test splits
│   ├── raft/               # RAFT train/validation splits with distractor passages
│   └── evaluation/         # Factual evaluation and abstention benchmarks
└── indices/                # Persisted search and retrieval indices
    ├── bm25/               # Rank-BM25 inverted index and tokenized vocabulary
    ├── faiss/              # FAISS dense vector indices (Flat, IVF, HNSW)
    └── docstore/           # Key-value passage store (passage_id -> passage text + metadata)
```

---

## Data Formats and Schemas

All processed training and evaluation data files are formatted as line-delimited JSON (`.jsonl`).

### 1. Document Passage Chunk (`Passage`)

Stored in `data/processed/chunks/passages.jsonl`:

```json
{
  "passage_id": "guideline_aha_hf_2023_p042_c003",
  "doc_id": "guideline_aha_hf_2023",
  "text": "In patients with HFrEF (LVEF <= 40%) and NYHA class II to IV symptoms, the use of an SGLT2 inhibitor (dapagliflozin or empagliflozin) is recommended to reduce the risk of CV death and hospitalization for heart failure, irrespective of the presence of type 2 diabetes (Class 1, Level of Evidence A).",
  "section": "Guideline-Directed Medical Therapy - SGLT2 Inhibitors",
  "page": 42,
  "chunk_index": 3,
  "metadata": {
    "doc_id": "guideline_aha_hf_2023",
    "title": "2023 ACC/AHA/HFSA Guideline for the Management of Heart Failure",
    "source": "Journal of the American College of Cardiology",
    "publication_date": "2023-04-01",
    "license": "Open Access / Guideline",
    "version": "1.0"
  }
}
```

### 2. SFT Example (`TrainingSFTExample`)

Stored in `data/processed/sft/train.jsonl`:

```json
{
  "question": "What is the recommended role of SGLT2 inhibitors in heart failure with reduced ejection fraction?",
  "context": "[Passage guideline_aha_hf_2023_p042_c003]: In patients with HFrEF (LVEF <= 40%) and NYHA class II to IV symptoms, the use of an SGLT2 inhibitor (dapagliflozin or empagliflozin) is recommended to reduce the risk of CV death and hospitalization for heart failure, irrespective of diabetes status.",
  "answer": "SGLT2 inhibitors (such as dapagliflozin and empagliflozin) are strongly recommended for patients with heart failure with reduced ejection fraction (HFrEF, LVEF <= 40%) and NYHA class II-IV symptoms to decrease cardiovascular mortality and heart failure hospitalizations [guideline_aha_hf_2023_p042_c003]. This recommendation applies regardless of whether the patient has type 2 diabetes.",
  "citation_ids": ["guideline_aha_hf_2023_p042_c003"],
  "answerability": "answerable"
}
```

### 3. RAFT Example (`TrainingRAFTExample`)

Stored in `data/processed/raft/train.jsonl`. RAFT training includes both relevant gold passages and distractor passages to teach the model to attend to relevant facts and ignore irrelevant context.

```json
{
  "question": "What is the first-line medication for uncomplicated hypertension in non-black patients according to clinical guidelines?",
  "relevant_passages": [
    {
      "passage_id": "guideline_acc_htn_p018_c001",
      "doc_id": "guideline_acc_htn",
      "text": "For initial therapy of hypertension in non-black patients, first-line agents include thiazide diuretics, calcium channel blockers (CCBs), and ACE inhibitors or ARBs.",
      "section": "Initial Pharmacologic Treatment",
      "page": 18,
      "chunk_index": 1
    }
  ],
  "distractor_passages": [
    {
      "passage_id": "guideline_asthma_gina_p030_c004",
      "doc_id": "guideline_asthma_gina",
      "text": "Low dose ICS-formoterol is the preferred reliever therapy across GINA Steps 1 and 2.",
      "section": "Stepwise Management of Asthma",
      "page": 30,
      "chunk_index": 4
    },
    {
      "passage_id": "guideline_copd_gold_p022_c002",
      "doc_id": "guideline_copd_gold",
      "text": "Initial pharmacological management for Group E includes LABA+LAMA combinations; consider LABA+LAMA+ICS if blood eosinophils >= 300 cells/uL.",
      "section": "COPD Pharmacotherapy",
      "page": 22,
      "chunk_index": 2
    }
  ],
  "answer": "According to clinical guidelines, first-line agents for initial monotherapy or combination therapy in non-black patients with uncomplicated hypertension include thiazide diuretics, calcium channel blockers (CCBs), and ACE inhibitors or ARBs [guideline_acc_htn_p018_c001]. Beta-blockers are not considered first-line unless there are specific compelling cardiovascular indications.",
  "supporting_passage_ids": ["guideline_acc_htn_p018_c001"],
  "answerability": "answerable"
}
```

### 4. Benchmark Evaluation Record (`DatasetExample`)

Stored in `data/processed/evaluation/test.jsonl`:

```json
{
  "question_id": "medqa_usmle_q0492",
  "question": "A 54-year-old male presents with persistent cough and exertional dyspnea. What diagnostic test is gold standard?",
  "reference_answer": "High-resolution computed tomography (HRCT) of the chest.",
  "gold_passage": "HRCT of the chest is considered the reference gold standard for diagnosing interstitial lung patterns...",
  "gold_passage_id": "passage_pulm_dx_c012",
  "doc_id": "pulmonology_dx_guide",
  "page": 14,
  "specialty": "Pulmonology",
  "difficulty": "medium",
  "answerability": "answerable",
  "source_url": "https://doi.org/10.xxxx/yyyy",
  "review_status": "clinically_reviewed"
}
```

---

## Data Ingestion Best Practices

1. **Section and Boundary Preservation**: Chunking must avoid breaking sentences or clinical numerical ranges (e.g., LVEF <= 40%).
2. **Citation Traceability**: Every generated claim must link to an explicit `passage_id`.
3. **PHI and De-identification**: No Protected Health Information (PHI) should ever be committed or processed in this directory. Ensure all clinical guidelines and papers comply with appropriate open-access or organizational licenses.
4. **Git Tracking**: Raw PDF/HTML data and binary vector index files (`.faiss`) are ignored by `.gitignore`.
