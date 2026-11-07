"""Unit tests for MedFusion-RAFT model providers, configurations, and factory."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.models import (
    AnswerabilityStatus,
    ClaimVerificationStatus,
    GenerationOutput,
    MedGemmaProvider,
    MockGeneratorProvider,
    MockVerifierProvider,
    ModelConfig,
    ModelError,
    OverallAssessmentStatus,
    ParsedMedicalAnswer,
    ParsedVerificationResult,
    QwenVerifierProvider,
    create_generator,
    create_verifier,
    register_generator_provider,
    register_verifier_provider,
)


def test_model_config_serialization() -> None:
    """Test ModelConfig default values, serialization, and YAML roundtrip."""
    config = ModelConfig(
        model_id="test/medical-model",
        quantization_bits=4,
        temperature=0.2,
    )

    assert config.model_id == "test/medical-model"
    assert config.quantization_bits == 4
    assert config.compute_dtype == "bfloat16"
    assert config.temperature == 0.2

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        config.to_yaml(tmp_path)
        loaded = ModelConfig.from_yaml(tmp_path)
        assert loaded.model_id == config.model_id
        assert loaded.quantization_bits == config.quantization_bits
        assert loaded.temperature == config.temperature
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_mock_generator_lifecycle_and_generation() -> None:
    """Test MockGeneratorProvider load, unload, and answer generation."""
    provider = MockGeneratorProvider(mode="realistic")
    assert not provider.is_loaded

    with provider:
        assert provider.is_loaded
        question = "What is the primary first-line therapy for type 2 diabetes?"
        context = ["[P1] Metformin is the established first-line oral hypoglycemic agent."]

        output = provider.generate_answer(question=question, context=context)
        assert isinstance(output, GenerationOutput)
        assert "ANSWER:" in output.text
        assert "CITATIONS:" in output.text
        assert "[P1]" in output.text

        parsed_data = output.metadata.get("parsed_answer")
        assert parsed_data is not None
        assert parsed_data["answerability"] == AnswerabilityStatus.ANSWERABLE.value
        assert "[P1]" in parsed_data["citations"]
        assert parsed_data["confidence"] >= 0.9

    assert not provider.is_loaded


def test_mock_generator_insufficient_evidence() -> None:
    """Test MockGeneratorProvider behavior with insufficient evidence."""
    provider = MockGeneratorProvider(mode="insufficient")
    with provider:
        output = provider.generate_answer(
            question="Can drug X cure condition Y?",
            context=[],
        )
        assert "INSUFFICIENT_EVIDENCE" in output.text
        parsed = output.metadata["parsed_answer"]
        assert parsed["answerability"] == AnswerabilityStatus.UNANSWERABLE.value


def test_mock_verifier_lifecycle_and_verification() -> None:
    """Test MockVerifierProvider verification modes and structured output."""
    provider = MockVerifierProvider(mode="realistic")
    with provider:
        assert provider.is_loaded
        question = "Is Metformin safe in mild renal impairment?"
        answer = "Metformin is relatively safe in mild renal impairment with dose monitoring."
        evidence = ["[P1] Metformin can be used safely in patients with mild renal impairment."]

        output = provider.verify_answer(question, answer, evidence)
        assert "CLAIMS:" in output.text
        assert "OVERALL_ASSESSMENT: VERIFIED" in output.text
        assert "SUPPORT_SCORE: 1.00" in output.text

        parsed = output.metadata.get("parsed_verification")
        assert parsed is not None
        assert parsed["overall_assessment"] == OverallAssessmentStatus.VERIFIED.value
        assert parsed["support_score"] == 1.0
        assert len(parsed["claims"]) >= 1
        assert parsed["claims"][0]["status"] == ClaimVerificationStatus.SUPPORTED.value

    assert not provider.is_loaded


def test_mock_verifier_reject_mode() -> None:
    """Test MockVerifierProvider rejection mode."""
    provider = MockVerifierProvider(mode="always_reject")
    output = provider.verify_answer(
        question="Query",
        answer="Ungrounded clinical claim",
        evidence=["[P1] Irrelevant context"],
    )
    parsed = output.metadata["parsed_verification"]
    assert parsed["overall_assessment"] == OverallAssessmentStatus.NOT_VERIFIED.value
    assert parsed["support_score"] == 0.0
    assert len(parsed["issues"]) > 0


def test_medgemma_output_parsing() -> None:
    """Test MedGemma regex parsing of structured answer format."""
    sample_response = (
        "ANSWER: Metformin reduces hepatic glucose production and increases insulin sensitivity [P1]. "
        "Lifestyle modifications should also be instituted [P2].\n"
        "CITATIONS: [P1], [P2]\n"
        "ANSWERABILITY: answerable\n"
        "CONFIDENCE: 0.96"
    )

    parsed = MedGemmaProvider.parse_medical_answer(sample_response)
    assert isinstance(parsed, ParsedMedicalAnswer)
    assert "Metformin reduces hepatic" in parsed.answer
    assert parsed.citations == ["[P1]", ["[P2]"]] or "[P1]" in parsed.citations
    assert parsed.answerability == AnswerabilityStatus.ANSWERABLE
    assert parsed.confidence == 0.96


def test_qwen_verifier_output_parsing() -> None:
    """Test Qwen verifier regex parsing of atomic claims, status, and assessment."""
    sample_response = (
        "CLAIMS:\n"
        "- Claim: Metformin lowers hepatic glucose | Status: SUPPORTED | Evidence: [P1] | Reasoning: Directly stated in P1\n"
        "- Claim: Aspirin is recommended | Status: UNSUPPORTED | Evidence: NONE | Reasoning: No mention in context\n\n"
        "OVERALL_ASSESSMENT: PARTIALLY_VERIFIED\n"
        "SUPPORT_SCORE: 0.50\n"
        "ISSUES: Second claim lacks passage citation"
    )

    parsed = QwenVerifierProvider.parse_verification_output(sample_response)
    assert isinstance(parsed, ParsedVerificationResult)
    assert len(parsed.claims) == 2
    assert parsed.claims[0].status == ClaimVerificationStatus.SUPPORTED
    assert parsed.claims[0].evidence == "[P1]"
    assert parsed.claims[1].status == ClaimVerificationStatus.UNSUPPORTED
    assert parsed.claims[1].evidence == "NONE"
    assert parsed.overall_assessment == OverallAssessmentStatus.PARTIALLY_VERIFIED
    assert parsed.support_score == 0.50
    assert len(parsed.issues) == 1


def test_model_factory() -> None:
    """Test create_generator and create_verifier factory resolution."""
    gen = create_generator(provider_type="mock")
    assert isinstance(gen, MockGeneratorProvider)

    ver = create_verifier(provider_type="mock")
    assert isinstance(ver, MockVerifierProvider)

    # Test dictionary configuration
    dict_cfg = {
        "model_id": "mock-test",
        "provider_type": "mock",
        "temperature": 0.5,
    }
    gen_dict = create_generator(dict_cfg)
    assert isinstance(gen_dict, MockGeneratorProvider)
    assert gen_dict.config.temperature == 0.5

    # Test unregistered provider raises ModelError
    with pytest.raises(ModelError):
        create_generator(provider_type="nonexistent_provider")


def test_factory_registry_extension() -> None:
    """Test registering custom generator and verifier providers."""

    class CustomGenerator(MockGeneratorProvider):
        pass

    register_generator_provider("custom_gen", CustomGenerator)
    gen = create_generator(provider_type="custom_gen")
    assert isinstance(gen, CustomGenerator)
