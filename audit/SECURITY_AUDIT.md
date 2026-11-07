# MedFusion-RAFT: Security Audit & Threat Modeling Report

**Audit Date**: 2026-11-08  
**Auditor**: Senior Security Auditor & Principal ML Engineer  
**Repository**: `medfusion-raft`  
**Classification**: Research Prototype AI Framework (Healthcare NLP Domain)  

---

## 1. Executive Security Assessment

MedFusion-RAFT is an experimental AI research framework. By design, it does not process protected health information (PHI) or personal identifiable information (PII). However, because it contains networked components (FastAPI backend, Streamlit frontend) and file deserialization pipelines, it must be evaluated against standard threat vectors.

### Overall Security Posture
- **Vulnerabilities Identified**: 4 (Low/Medium risk in research environments; Critical if deployed to public internet).
- **Hardcoded Secrets**: None found.
- **Leaked API Keys**: None found.
- **PHI / PII in Datasets**: None found.
- **Remote Code Execution (RCE) Vectors**: Python `pickle` deserialization on local index files (mitigated via documentation and trusted-source constraints).

---

## 2. Threat Vector Analysis & Audit Findings

### SEC-01: Insecure Object Deserialization via Python `pickle`
- **Severity**: **Medium** (High if exposed to untrusted uploads)
- **Location**:
  - `src/retrieval/bm25_retriever.py`: `BM25Retriever.load_index()`
  - `src/retrieval/dense_retriever.py`: `MedCPTRetriever.load_index()`
- **Analysis**:
  Both BM25 and dense retrieval indices are serialized using Python's standard `pickle` module. If a malicious user or attacker can replace the index file on disk or pass an arbitrary path to an untrusted `.pkl` file via an unauthenticated endpoint, Python's `pickle.load()` could execute arbitrary operating system commands.
- **Current Mitigation**:
  In the current architecture, `load_index()` is invoked only via trusted internal server scripts (`PipelineService.initialize()` and `scripts/evaluate.py`), and does not take user-uploaded binary files.
- **Remediation Implemented**:
  Added explicit security warnings in the method docstrings. For production deployments, recommended migration to `safetensors`, JSONL metadata, or Arrow/Parquet formats.

---

### SEC-02: Missing API Authentication & Authorization
- **Severity**: **Medium** (Acceptable for local research; Unacceptable for public network exposure)
- **Location**: `backend/app.py`, `backend/api/routes.py`
- **Analysis**:
  The FastAPI application defines endpoints (`/api/v1/ask`, `/api/v1/batch`, `/api/v1/evaluate`) with no API key, bearer token, or OAuth2 authentication middleware. Anyone with network access to the API port (default 8000) can submit questions, trigger inference, or initiate evaluations.
- **Remediation Recommendation**:
  For multi-user or networked deployments, add a FastAPI dependency enforcing `X-API-Key` or JWT validation:
  ```python
  from fastapi.security import APIKeyHeader
  api_key_header = APIKeyHeader(name="X-API-Key")
  ```

---

### SEC-03: Permissive CORS Configuration (Wildcard `*`)
- **Severity**: **Low** (Informational in local research environment)
- **Location**: `backend/app.py` lines 42-48
- **Analysis**:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
  `allow_origins=["*"]` allows any web origin to initiate cross-origin HTTP requests against the backend.
- **Remediation Recommendation**:
  Restrict allowed origins to the Streamlit frontend origin (e.g., `["http://localhost:8501"]` or environment-configured domain).

---

### SEC-04: Potential Path Traversal in Evaluation Endpoint
- **Severity**: **Low / Medium**
- **Location**: `backend/api/routes.py` line 134 (`POST /api/v1/evaluate`)
- **Analysis**:
  The `/evaluate` endpoint accepts a `dataset_path` string in `EvaluationRequest`. While `Path(request.dataset_path).is_file()` checks that the file exists, it does not restrict the path to a designated data directory, potentially allowing an attacker to confirm the existence of arbitrary system files.
- **Remediation Recommendation**:
  Enforce path sandboxing using `Path.resolve()` and verify that the target path is a descendant of the configured `data/` directory:
  ```python
  resolved = Path(request.dataset_path).resolve()
  if not resolved.is_relative_to(data_dir.resolve()):
      raise HTTPException(status_code=403, detail="Access outside data directory forbidden")
  ```

---

## 3. Data Privacy & PHI / PII Audit

A comprehensive scan was conducted across all committed datasets:
- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/processed/sft_train.jsonl`
- `data/processed/sft_validation.jsonl`
- `data/processed/raft_train.jsonl`
- `data/processed/raft_validation.jsonl`

### Findings:
1. **Zero Real Patient Data**: All QA pairs represent synthetic clinical guideline scenarios (e.g., first-line therapy for uncomplicated hypertension, metformin contraindications, asthma diagnosis).
2. **No Identifiable Attributes**: No patient names, dates of birth, medical record numbers (MRNs), hospital names, social security numbers, or geographic coordinates exist in any dataset file.
3. **Open-Source Compliance**: Public benchmark references (PubMedQA, MedQA) are properly cited with MIT/Apache-2.0 licenses.

---

## 4. Secret & Credential Scanning

A thorough scan for sensitive keys was performed across the repository:
- `HUGGING_FACE_HUB_TOKEN`: Present only as an empty placeholder variable in `.env.example` and `src/utils/config.py`. No actual tokens committed.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`: None found.
- Private SSH / GPG Keys: None found.
- `.gitignore`: Properly excludes `.env`, `*.pem`, `*.key`, `checkpoints/`, `models/cache/`, and `.pytest_cache/`.

---

## 5. Medical Safety & Disclaimer Notice

Because MedFusion-RAFT operates in the healthcare domain, medical safety is a paramount concern:
- The Streamlit interface displays a permanent, high-contrast banner:  
  `⚠️ RESEARCH PROTOTYPE - Not for clinical use or medical decision-making.`
- The README and research documentation repeatedly state that the system is an experimental AI research framework, has not been validated in clinical trials, and must never be used for diagnosis or treatment.
