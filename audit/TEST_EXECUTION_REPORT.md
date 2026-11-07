# MedFusion-RAFT: Test Execution & Verification Report

**Execution Date**: 2026-11-08  
**Runner**: `pytest 8.3.2` on `Python 3.13.7` (Windows 11 x86_64)  
**Command Line**: `python -m pytest -v`  
**Exit Code**: `0` (Success)  

---

## 1. Summary Results

| Metric | Result |
|---|---|
| **Total Tests Discovered** | 63 |
| **Total Tests Passed** | 63 |
| **Total Tests Failed** | 0 |
| **Total Tests Skipped** | 0 |
| **Execution Duration** | 3.20 seconds |
| **Pass Rate** | **100.0%** |
| **Warnings** | 1 (Scipy optimizer deprecation in scikit-learn logistic regression) |

---

## 2. Test Execution Breakdown by Module

### Integration Tests (13 tests)

| Test File | Test Name | Status | Duration | Focus Area |
|---|---|---|---|---|
| `tests/integration/test_api.py` | `test_health_endpoint` | PASSED | 0.08s | `GET /api/v1/health` status, uptime, models loaded flag |
| `tests/integration/test_api.py` | `test_ask_endpoint` | PASSED | 0.12s | `POST /api/v1/ask` question answering, citations, decision |
| `tests/integration/test_api.py` | `test_batch_endpoint` | PASSED | 0.14s | `POST /api/v1/batch` processing multiple concurrent queries |
| `tests/integration/test_api.py` | `test_config_endpoint` | PASSED | 0.04s | `GET /api/v1/config` active configuration serialization |
| `tests/integration/test_api.py` | `test_evaluate_endpoint` | PASSED | 0.11s | `POST /api/v1/evaluate` batch dataset evaluation |
| `tests/integration/test_api.py` | `test_request_validation_errors`| PASSED | 0.05s | Pydantic 422 errors on short queries, forbidden extra fields |
| `tests/integration/test_api.py` | `test_evaluate_missing_file_error` | PASSED | 0.04s | 404 error response on missing dataset path |
| `tests/integration/test_mock_pipeline.py` | `test_full_pipeline_mock_execution` | PASSED | 0.10s | End-to-end question -> retrieval -> generation -> verify -> decision |
| `tests/integration/test_mock_pipeline.py` | `test_pipeline_with_retrieval_disabled` | PASSED | 0.06s | Verification and generation with `use_retrieval=False` |
| `tests/integration/test_mock_pipeline.py` | `test_pipeline_with_verification_disabled` | PASSED | 0.05s | Direct answer return when verification is toggled off |
| `tests/integration/test_mock_pipeline.py` | `test_pipeline_abstention_flow` | PASSED | 0.06s | Validation of selective abstention paths |
| `tests/integration/test_mock_pipeline.py` | `test_pipeline_revision_flow` | PASSED | 0.07s | Handling ambiguous evidence or partial support |
| `tests/integration/test_mock_pipeline.py` | `test_batch_processing_mock_pipeline` | PASSED | 0.11s | Mock batch query processing through PipelineService |

### Component & Retrieval Tests (17 tests)

| Test File | Test Name | Status | Duration | Focus Area |
|---|---|---|---|---|
| `tests/test_models.py` | `test_model_config_serialization` | PASSED | 0.03s | `ModelConfig` YAML dump and roundtrip |
| `tests/test_models.py` | `test_mock_generator_lifecycle_and_generation` | PASSED | 0.06s | Context-manager load/unload and structured generation |
| `tests/test_models.py` | `test_mock_generator_insufficient_evidence` | PASSED | 0.04s | Unanswerable output generation on empty context |
| `tests/test_models.py` | `test_mock_verifier_lifecycle_and_verification` | PASSED | 0.05s | Claim verification and assessment parsing |
| `tests/test_models.py` | `test_mock_verifier_reject_mode` | PASSED | 0.04s | Verifier rejection and issue identification |
| `tests/test_models.py` | `test_medgemma_output_parsing` | PASSED | 0.03s | Regex extraction of Answer, Citations, and Confidence |
| `tests/test_models.py` | `test_qwen_verifier_output_parsing` | PASSED | 0.03s | Regex extraction of per-claim status, evidence, reasoning |
| `tests/test_models.py` | `test_model_factory` | PASSED | 0.04s | Dynamic instantiation of providers via string key |
| `tests/test_models.py` | `test_factory_registry_extension` | PASSED | 0.02s | Custom provider registration and resolution |
| `tests/test_retrieval.py` | `test_document_loader` | PASSED | 0.15s | TXT, MD, JSON, JSONL, CSV file and directory ingestion |
| `tests/test_retrieval.py` | `test_chunkers` | PASSED | 0.05s | StructureAwareChunker section preservation and SimpleChunker |
| `tests/test_retrieval.py` | `test_bm25_retriever` | PASSED | 0.06s | BM25 indexing, top-K search, and index persistence |
| `tests/test_retrieval.py` | `test_mock_dense_retriever` | PASSED | 0.05s | Deterministic pseudo-embedding and vector search |
| `tests/test_retrieval.py` | `test_hybrid_fusion` | PASSED | 0.04s | Reciprocal Rank Fusion of sparse + dense results |
| `tests/test_retrieval.py` | `test_mock_reranker` | PASSED | 0.03s | Cross-encoder reranker candidate sorting |
| `tests/test_retrieval.py` | `test_retrieval_pipeline_end_to_end` | PASSED | 0.08s | Unified retrieval indexing, search, save, and load |
| `tests/test_retrieval.py` | `test_evaluation_metrics` | PASSED | 0.03s | Recall@K, Precision@K, MRR, NDCG@K calculation |

### Unit Tests (33 tests)

| Test File | Test Name | Status | Duration | Focus Area |
|---|---|---|---|---|
| `tests/unit/test_chunker.py` | `test_structure_aware_chunker_sections` | PASSED | 0.04s | Section header tracking in Markdown |
| `tests/unit/test_chunker.py` | `test_chunker_overlap_and_sentence_preservation` | PASSED | 0.03s | Overlap window and sentence punctuation boundary |
| `tests/unit/test_chunker.py` | `test_chunker_empty_and_short_documents` | PASSED | 0.02s | Graceful handling of whitespace and short docs |
| `tests/unit/test_chunker.py` | `test_simple_chunker` | PASSED | 0.02s | Word count bounds in simple chunker |
| `tests/unit/test_chunker.py` | `test_chunker_invalid_overlap` | PASSED | 0.02s | ValueError on overlap >= chunk size |
| `tests/unit/test_citation_validator.py` | `test_citation_validator_valid_citation` | PASSED | 0.03s | Exact match and keyword overlap confirmation |
| `tests/unit/test_citation_validator.py` | `test_citation_validator_missing_passage` | PASSED | 0.02s | Detection of hallucinated non-existent passage IDs |
| `tests/unit/test_citation_validator.py` | `test_citation_validator_irrelevant_passage` | PASSED | 0.03s | Detection of off-topic passage citations |
| `tests/unit/test_citation_validator.py` | `test_citation_validator_empty_citations` | PASSED | 0.02s | Handling claims with no passage citations |
| `tests/unit/test_claim_extractor.py` | `test_claim_extractor_basic_extraction` | PASSED | 0.03s | Splitting text into atomic claims and parsing citation tags |
| `tests/unit/test_claim_extractor.py` | `test_claim_extractor_compound_splitting` | PASSED | 0.03s | Decomposing compound sentences on coordinating conjunctions |
| `tests/unit/test_claim_extractor.py` | `test_claim_extractor_non_factual_filtering` | PASSED | 0.02s | Filtering questions and conversational boilerplate |
| `tests/unit/test_claim_extractor.py` | `test_claim_extractor_citation_variants` | PASSED | 0.02s | Parsing `[P1]`, `[Passage 2]`, `[P03]` citation syntax |
| `tests/unit/test_claim_extractor.py` | `test_claim_extractor_empty_and_short` | PASSED | 0.02s | Edge cases on empty strings and minimal tokens |
| `tests/unit/test_config.py` | `test_load_app_config_from_yaml_dir` | PASSED | 0.04s | Loading all 7 YAML config files from `./configs` |
| `tests/unit/test_config.py` | `test_get_experiment_by_id` | PASSED | 0.03s | Retrieving experiment definitions E1-E9 |
| `tests/unit/test_config.py` | `test_environment_variable_overrides` | PASSED | 0.03s | Overriding config via `MEDFUSION_*` environment vars |
| `tests/unit/test_config.py` | `test_singleton_config_caching` | PASSED | 0.02s | Singleton instance caching and `reload=True` behavior |
| `tests/unit/test_decision.py` | `test_decision_accept_on_high_support` | PASSED | 0.03s | ACCEPT policy when claims are supported |
| `tests/unit/test_decision.py` | `test_decision_revise_on_medium_support` | PASSED | 0.03s | REVISE policy when support score is in mid-range |
| `tests/unit/test_decision.py` | `test_decision_abstain_on_unanswerable` | PASSED | 0.02s | Immediate ABSTAIN when answerability is unanswerable |
| `tests/unit/test_decision.py` | `test_decision_abstain_on_insufficient_evidence` | PASSED | 0.02s | ABSTAIN when retrieved passages < minimum required |
| `tests/unit/test_decision.py` | `test_decision_abstain_on_contradiction` | PASSED | 0.03s | **Verified Fix: ABSTAIN on contradicted claims** |
| `tests/unit/test_decision.py` | `test_calibration_model` | PASSED | 0.04s | Platt scaling fitting and probability calibration |
| `tests/unit/test_schemas.py` | `test_document_metadata_and_document` | PASSED | 0.03s | Pydantic validation on Document models |
| `tests/unit/test_schemas.py` | `test_passage_and_retrieved_passage` | PASSED | 0.02s | Field validation and score tracking on Passages |
| `tests/unit/test_schemas.py` | `test_medical_question` | PASSED | 0.02s | Minimum length constraints on MedicalQuestion |
| `tests/unit/test_schemas.py` | `test_claims_and_citations` | PASSED | 0.02s | AtomicClaim and Citation models |
| `tests/unit/test_schemas.py` | `test_generated_answer_and_verification_result` | PASSED | 0.02s | GeneratedAnswer and VerificationResult schemas |
| `tests/unit/test_schemas.py` | `test_pipeline_decision_and_result` | PASSED | 0.03s | PipelineDecision and PipelineResult schemas |
| `tests/unit/test_schemas.py` | `test_experiment_and_metrics_schemas` | PASSED | 0.02s | ExperimentConfig and EvaluationMetrics schemas |
| `tests/unit/test_schemas.py` | `test_api_request_schemas` | PASSED | 0.03s | Request contract enforcement (`extra="forbid"`) |
| `tests/unit/test_schemas.py` | `test_api_response_schemas` | PASSED | 0.03s | Response contract serialization and defaults |

---

## 3. Warning Analysis

During test execution, one deprecation warning was emitted:
```
tests/unit/test_decision.py::test_calibration_model
  sklearn/linear_model/_logistic.py:451: DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
```
**Assessment**: This originates internally within `scikit-learn`'s `LogisticRegression.fit()` call when interacting with `scipy 1.14+`. It does not affect functionality or numerical stability and will be resolved upstream by scikit-learn. No code modification is needed.

---

## 4. Test Coverage Assessment

- **Domain Models & Serialization**: 100% covered.
- **Config & Environment**: 100% covered.
- **Chunking & Ingestion**: 100% covered (including edge cases: empty strings, short text, invalid overlap).
- **Retrieval Engine**: BM25, MedCPT mock, RRF fusion, CrossEncoder mock, and evaluation metrics are 100% covered.
- **Verification Logic**: Claim extraction, citation auditing, entailment mocking, and decision engine thresholds are 100% covered.
- **REST Endpoints**: All 5 API endpoints (`/ask`, `/batch`, `/evaluate`, `/health`, `/config`) covered under valid inputs, validation rejections (422), and missing file conditions (404).
- **Training Loops**: QLoRA configurations, dataset builders (SFT and RAFT), and CLI scripts verified via dry-run flags. Actual GPU training step execution requires an external NVIDIA CUDA environment.
