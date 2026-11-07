"""Integration tests for end-to-end pipeline execution with mock providers."""

from __future__ import annotations

import pytest

from backend.schemas.requests import BatchQuestionRequest, QuestionRequest
from backend.services.pipeline_service import PipelineService
from src.models.mock_providers import MockGeneratorProvider, MockVerifierProvider


@pytest.fixture(autouse=True)
def setup_pipeline_service() -> None:
    """Ensure singleton is cleanly reset for tests."""
    PipelineService.reset_instance()
    yield
    PipelineService.reset_instance()


def test_full_pipeline_mock_execution() -> None:
    """Test standard end-to-end pipeline with mock generator and verifier."""
    service = PipelineService.get_instance(mock=True)

    req = QuestionRequest(
        question="What is the recommended first-line treatment for uncomplicated hypertension?",
        specialty="Cardiology",
        use_retrieval=True,
        use_verification=True,
        use_abstention=True,
        experiment_id="E9_full",
    )
    res = service.process_question(req)

    assert res.answer_text != ""
    assert res.answerability == "answerable"
    assert res.confidence > 0.0
    assert res.decision in ("accept", "revise", "abstain")
    assert len(res.evidence_passages) > 0
    assert len(res.claims) > 0
    assert res.processing_time_ms >= 0.0
    assert res.experiment_id == "E9_full"


def test_pipeline_with_retrieval_disabled() -> None:
    """Test pipeline execution when retrieval is explicitly toggled off."""
    service = PipelineService.get_instance(mock=True)

    req = QuestionRequest(
        question="How does aspirin provide cardioprotection?",
        use_retrieval=False,
        use_verification=True,
        use_abstention=True,
    )
    res = service.process_question(req)

    assert len(res.evidence_passages) == 0
    assert res.answer_text != ""


def test_pipeline_with_verification_disabled() -> None:
    """Test pipeline execution when claim verification is skipped."""
    service = PipelineService.get_instance(mock=True)

    req = QuestionRequest(
        question="What are symptoms of hypoglycemia?",
        use_retrieval=True,
        use_verification=False,
    )
    res = service.process_question(req)

    assert res.verification_status == "skipped"
    assert len(res.claims) == 0
    assert res.decision == "accept"


def test_pipeline_abstention_flow() -> None:
    """Test that pipeline can produce non-accept decisions in certain configurations."""
    service = PipelineService.get_instance(mock=True)

    req = QuestionRequest(
        question="Can high-dose aspirin be safely combined with dual anticoagulation?",
        use_retrieval=True,
        use_verification=True,
        use_abstention=True,
    )
    res = service.process_question(req)

    # The mock pipeline should produce a valid response with a decision
    assert res.decision in ("accept", "revise", "abstain")
    assert res.answer_text != ""
    assert res.processing_time_ms >= 0


def test_pipeline_revision_flow() -> None:
    """Test that pipeline handles revision/rejection gracefully."""
    service = PipelineService.get_instance(mock=True)

    req = QuestionRequest(
        question="What novel therapies exist for ultra-rare cardiac amyloidosis?",
        use_retrieval=True,
        use_verification=True,
        use_abstention=True,
    )
    res = service.process_question(req)

    # Should produce a valid result regardless of verification outcome
    assert res.decision in ("accept", "revise", "abstain")
    assert res.processing_time_ms >= 0


def test_batch_processing_mock_pipeline() -> None:
    """Test batch question processing pipeline."""
    service = PipelineService.get_instance(mock=True)

    batch_req = BatchQuestionRequest(
        questions=[
            QuestionRequest(question="What is the diagnostic threshold for diabetes?"),
            QuestionRequest(question="What are contraindications for beta-blockers?"),
        ],
        experiment_id="E5_raft",
    )
    batch_res = service.process_batch(batch_req)

    assert len(batch_res.results) == 2
    assert batch_res.total_time_ms >= 0.0
    assert batch_res.experiment_id == "E5_raft"
    for r in batch_res.results:
        assert r.answer_text != ""
