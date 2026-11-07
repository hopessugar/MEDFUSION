#!/usr/bin/env python3
"""Run the MedFusion-RAFT pipeline on a single question or interactively.

Usage:
    python scripts/run_pipeline.py --mock --question "What is metformin used for?"
    python scripts/run_pipeline.py --mock --interactive
    python scripts/run_pipeline.py --config-dir ./configs --question "..."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_mock_pipeline():
    """Create a pipeline using mock providers for CPU testing."""
    from src.models.mock_providers import MockGeneratorProvider, MockVerifierProvider
    from src.models.base import ModelConfig
    from src.verification.claim_extractor import ClaimExtractor
    from src.verification.citation_validator import CitationValidator
    from src.verification.entailment import MockEntailmentChecker
    from src.verification.pipeline import VerificationPipeline, VerificationPipelineConfig
    from src.verification.decision import DecisionEngine, DecisionPolicyConfig
    from src.models.schemas import (
        MedicalQuestion,
        Passage,
        RetrievedPassage,
        GeneratedAnswer,
        AnswerabilityLabel,
    )

    # Create mock generator
    gen_config = ModelConfig(model_id="mock-medgemma", quantization_bits=None)
    generator = MockGeneratorProvider(gen_config)
    generator.load()

    # Create mock verifier
    ver_config = ModelConfig(model_id="mock-qwen-verifier", quantization_bits=None)
    verifier = MockVerifierProvider(ver_config)
    verifier.load()

    # Create verification pipeline
    ver_pipeline_config = VerificationPipelineConfig(
        use_llm_verification=False,  # Skip LLM verification in mock mode
    )
    ver_pipeline = VerificationPipeline(
        config=ver_pipeline_config,
        verifier_provider=verifier,
        entailment_checker=MockEntailmentChecker(),
    )

    # Create decision engine
    decision_engine = DecisionEngine(DecisionPolicyConfig())

    return generator, verifier, ver_pipeline, decision_engine


def run_single_question(question_text: str, use_mock: bool = True) -> dict:
    """Run the pipeline on a single question."""
    from src.models.schemas import (
        MedicalQuestion,
        RetrievedPassage,
        GeneratedAnswer,
        AnswerabilityLabel,
    )

    print(f"\n{'='*60}")
    print(f"Question: {question_text}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()

    if use_mock:
        generator, verifier, ver_pipeline, decision_engine = create_mock_pipeline()

        # Create mock retrieved passages
        passages = [
            RetrievedPassage(
                passage_id="passage_0001",
                doc_id="doc_001",
                text=(
                    "Metformin is a biguanide oral antidiabetic agent used as "
                    "first-line treatment for type 2 diabetes mellitus. It works "
                    "by decreasing hepatic glucose production and increasing "
                    "insulin sensitivity in peripheral tissues."
                ),
                section="Pharmacology",
                score=0.92,
                retrieval_method="hybrid_rrf",
                rank=1,
            ),
            RetrievedPassage(
                passage_id="passage_0002",
                doc_id="doc_001",
                text=(
                    "Contraindications for metformin include severe renal "
                    "impairment (eGFR < 30 mL/min), acute or chronic metabolic "
                    "acidosis, and hypersensitivity to the drug. Caution is "
                    "advised in hepatic impairment."
                ),
                section="Contraindications",
                score=0.85,
                retrieval_method="hybrid_rrf",
                rank=2,
            ),
            RetrievedPassage(
                passage_id="passage_0003",
                doc_id="doc_002",
                text=(
                    "Common side effects of metformin include gastrointestinal "
                    "symptoms such as nausea, diarrhea, and abdominal discomfort. "
                    "These can be minimized by starting with a low dose and "
                    "titrating gradually."
                ),
                section="Adverse Effects",
                score=0.78,
                retrieval_method="hybrid_rrf",
                rank=3,
            ),
        ]

        # Generate answer
        print("🔧 Step 1: Generating answer with mock MedGemma...")
        context = "\n\n".join(
            f"[P{i+1}] {p.text}" for i, p in enumerate(passages)
        )
        gen_output = generator.generate_answer(question_text, context)
        print(f"   Generated {gen_output.output_tokens} tokens in {gen_output.generation_time_ms:.0f}ms")

        # Parse into GeneratedAnswer
        answer = GeneratedAnswer(
            answer_text=gen_output.text.split("ANSWER:")[-1].split("CITATIONS:")[0].strip()
            if "ANSWER:" in gen_output.text else gen_output.text,
            answerability=AnswerabilityLabel.ANSWERABLE,
            confidence=0.8,
            model_id="mock-medgemma",
            generation_time_ms=gen_output.generation_time_ms,
        )

        # Verify
        print("\n🔍 Step 2: Verifying answer...")
        ver_result = ver_pipeline.verify(question_text, answer, passages)
        print(f"   Verified: {ver_result.verified}")
        print(f"   Supported: {ver_result.supported_claims}, Unsupported: {ver_result.unsupported_claims}")

        # Decision
        print("\n⚖️ Step 3: Making decision...")
        decision = decision_engine.decide(
            answer=answer,
            verification=ver_result,
            num_evidence_passages=len(passages),
        )
        print(f"   Decision: {decision.decision.value.upper()}")
        print(f"   Score: {decision.score:.3f}")
        print(f"   Reasoning: {decision.reasoning}")

        elapsed = (time.perf_counter() - start_time) * 1000

        result = {
            "question": question_text,
            "answer": answer.answer_text,
            "answerability": answer.answerability.value,
            "confidence": answer.confidence,
            "decision": decision.decision.value,
            "decision_score": decision.score,
            "decision_reasoning": decision.reasoning,
            "verification_verified": ver_result.verified,
            "verification_score": ver_result.overall_score,
            "supported_claims": ver_result.supported_claims,
            "unsupported_claims": ver_result.unsupported_claims,
            "contradicted_claims": ver_result.contradicted_claims,
            "num_evidence_passages": len(passages),
            "total_time_ms": elapsed,
        }

        print(f"\n{'='*60}")
        print(f"📋 FINAL ANSWER ({decision.decision.value.upper()})")
        print(f"{'='*60}")
        print(f"\n{answer.answer_text}")
        print(f"\n⏱️ Total time: {elapsed:.0f}ms")

        return result
    else:
        print("Full pipeline with real models requires GPU and model access.")
        print("Use --mock flag for CPU testing.")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run the MedFusion-RAFT pipeline",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Medical question to answer",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock model providers (no GPU needed)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./configs",
        help="Path to configuration directory",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save result to JSON file",
    )

    args = parser.parse_args()

    if args.interactive:
        print("\n🔬 MedFusion-RAFT Interactive Mode")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not question:
                    continue
                run_single_question(question, use_mock=args.mock)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    elif args.question:
        result = run_single_question(args.question, use_mock=args.mock)
        if args.output and result:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Result saved to {args.output}")
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python scripts/run_pipeline.py --mock -q "What is metformin used for?"')
        print("  python scripts/run_pipeline.py --mock --interactive")


if __name__ == "__main__":
    main()

