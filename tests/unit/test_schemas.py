"""Unit tests for core Pydantic data schemas and API models."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from backend.schemas.requests import (
    BatchQuestionRequest,
    EvaluationRequest,
    QuestionRequest,
)
from backend.schemas.responses import (
    AnswerResponse,
    BatchAnswerResponse,
    ClaimResponse,
    ErrorResponse,
    HealthResponse,
    PassageResponse,
)
from src.models.schemas import (
    AnswerabilityLabel,
    AtomicClaim,
    Citation,
    ClaimSupportStatus,
    DecisionType,
    Document,
    DocumentMetadata,
    EvaluationMetrics,
    ExperimentConfig,
    ExperimentResult,
    GeneratedAnswer,
    MedicalQuestion,
    Passage,
    PipelineDecision,
    PipelineResult,
    QuestionDifficulty,
    RetrievedPassage,
    TrainingRAFTExample,
    TrainingSFTExample,
    VerificationResult,
)


def test_document_metadata_and_document() -> None:
    """Test Document and DocumentMetadata validation and serialization."""
    meta = DocumentMetadata(
        doc_id="doc_01",
        title="Hypertension Practice Guidelines",
        source="AHA/ACC",
        url="https://doi.org/10.1161/example",
    )
    assert meta.doc_id == "doc_01"
    assert meta.title == "Hypertension Practice Guidelines"

    doc = Document(
        doc_id="doc_01",
        title="Hypertension Practice Guidelines",
        text="Full document content describing clinical protocols.",
        metadata={"year": 2024, "source": "AHA/ACC"},
    )
    assert doc.doc_id == "doc_01"
    assert "year" in doc.metadata

    # Validation failure on empty text
    with pytest.raises(ValidationError):
        Document(doc_id="doc_02", text="")


def test_passage_and_retrieved_passage() -> None:
    """Test Passage and RetrievedPassage chunk models."""
    passage = Passage(
        passage_id="p_001",
        doc_id="doc_01",
        text="Beta-blockers are second-line agents for hypertension.",
        section="Pharmacology",
        chunk_index=0,
    )
    assert passage.passage_id == "p_001"
    assert passage.metadata == {}

    retrieved = RetrievedPassage(
        passage_id="p_001",
        doc_id="doc_01",
        text="Beta-blockers are second-line agents for hypertension.",
        score=0.895,
        retrieval_method="rrf_hybrid",
        rank=1,
    )
    assert retrieved.score == 0.895
    assert retrieved.retrieval_method == "rrf_hybrid"
    assert retrieved.rank == 1

    dumped = retrieved.model_dump()
    assert dumped["score"] == 0.895


def test_medical_question() -> None:
    """Test MedicalQuestion schema constraints."""
    q = MedicalQuestion(
        question_id="q_100",
        question="What are contraindications for ACE inhibitors?",
        specialty="Cardiology",
        difficulty=QuestionDifficulty.MEDIUM,
        expected_answerability=AnswerabilityLabel.ANSWERABLE,
    )
    assert q.question_id == "q_100"
    assert q.difficulty == QuestionDifficulty.MEDIUM

    # Question text min length constraint (< 3 chars should fail)
    with pytest.raises(ValidationError):
        MedicalQuestion(question_id="q_err", question="No")


def test_claims_and_citations() -> None:
    """Test AtomicClaim and Citation models."""
    cit = Citation(claim_text="ACE inhibitors cause cough.", passage_id="p_001", relevance_score=0.92)
    assert cit.passage_id == "p_001"
    assert cit.relevance_score == 0.92

    claim = AtomicClaim(
        claim_id="c_01",
        claim_text="ACE inhibitors commonly cause a dry cough.",
        citations=[cit],
        support_status=ClaimSupportStatus.SUPPORTED,
        entailment_score=0.98,
    )
    assert claim.support_status == ClaimSupportStatus.SUPPORTED
    assert len(claim.citations) == 1

    # Claim text min length constraint
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="c_err", claim_text="AB")


def test_generated_answer_and_verification_result() -> None:
    """Test GeneratedAnswer and VerificationResult schemas."""
    ans = GeneratedAnswer(
        answer_text="Lisinopril is an ACE inhibitor used for hypertension.",
        answerability=AnswerabilityLabel.ANSWERABLE,
        confidence=0.95,
        generation_time_ms=120.5,
    )
    assert ans.confidence == 0.95
    assert ans.answerability == AnswerabilityLabel.ANSWERABLE

    ver = VerificationResult(
        verified=True,
        overall_score=0.95,
        supported_claims=3,
        unsupported_claims=0,
        contradicted_claims=0,
    )
    assert ver.verified is True
    assert ver.supported_claims == 3


def test_pipeline_decision_and_result() -> None:
    """Test PipelineDecision and PipelineResult lifecycle schemas."""
    dec = PipelineDecision(
        decision=DecisionType.ACCEPT,
        score=0.92,
        reasoning="All atomic propositions verified by guideline passages.",
    )
    assert dec.decision == DecisionType.ACCEPT
    assert dec.score == 0.92

    q = MedicalQuestion(question_id="q_1", question="What treats diabetic ketoacidosis?")
    res = PipelineResult(
        question=q,
        decision=dec,
        final_answer="Intravenous regular insulin and fluid replacement.",
        total_time_ms=350.2,
    )
    assert res.decision.decision == DecisionType.ACCEPT
    assert res.total_time_ms == 350.2
    assert res.timestamp is not None


def test_experiment_and_metrics_schemas() -> None:
    """Test ExperimentConfig and EvaluationMetrics schemas."""
    cfg = ExperimentConfig(
        experiment_id="E5_raft",
        name="RAFT Retrieval-Augmented Fine-Tuning",
        generator="raft",
        retrieval="hybrid",
        fine_tuning="raft",
        verification=True,
        abstention=True,
    )
    assert cfg.experiment_id == "E5_raft"
    assert cfg.verification is True

    metrics = EvaluationMetrics(
        accuracy=0.88,
        f1=0.84,
        claim_support_rate=0.92,
        unsupported_claim_rate=0.05,
    )
    assert metrics.accuracy == 0.88
    assert metrics.claim_support_rate == 0.92

    exp_result = ExperimentResult(config=cfg, metrics=metrics, num_examples=100, status="COMPLETED")
    assert exp_result.status == "COMPLETED"
    assert exp_result.num_examples == 100


def test_api_request_schemas() -> None:
    """Test FastAPI QuestionRequest, BatchQuestionRequest, and EvaluationRequest."""
    req = QuestionRequest(
        question="What is the first-line medication for asthma?",
        specialty="Pulmonology",
        use_retrieval=True,
        use_verification=True,
        use_abstention=True,
    )
    assert req.use_retrieval is True
    assert req.specialty == "Pulmonology"

    # Extra attributes forbidden
    with pytest.raises(ValidationError):
        QuestionRequest(question="Valid question text", unknown_field=123)  # type: ignore

    batch = BatchQuestionRequest(questions=[req])
    assert len(batch.questions) == 1

    eval_req = EvaluationRequest(dataset_path="data/test.json", experiment_id="E5_raft")
    assert eval_req.dataset_path == "data/test.json"


def test_api_response_schemas() -> None:
    """Test API response models serialization and defaults."""
    p_resp = PassageResponse(passage_id="p1", text="Sample guideline text", score=0.91)
    c_resp = ClaimResponse(claim_text="Sample claim", support_status="supported", evidence_passage_id="p1")
    ans_resp = AnswerResponse(
        answer_text="Sample answer text.",
        citations=["[P1]"],
        confidence=0.95,
        claims=[c_resp],
        verification_status="verified",
        decision="accept",
        evidence_passages=[p_resp],
        processing_time_ms=45.2,
    )

    dumped = ans_resp.model_dump()
    assert dumped["decision"] == "accept"
    assert len(dumped["claims"]) == 1

    health = HealthResponse(status="healthy", models_loaded=True, uptime_seconds=12.5)
    assert health.status == "healthy"

    err = ErrorResponse(error="Pipeline Error", detail="Inference timed out")
    assert err.error == "Pipeline Error"
    assert err.timestamp is not None
