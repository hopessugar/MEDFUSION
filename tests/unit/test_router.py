"""Unit tests for the RuleBasedRouter and AdaptiveRouter."""

from __future__ import annotations

import pytest

from src.routing.router import (
    AdaptiveRouter,
    QuestionComplexity,
    RouterConfig,
    RoutingDecision,
    RuleBasedRouter,
)


def test_router_critical_query() -> None:
    """Test safety-critical inquiries routed to EVIDENCE_CRITICAL."""
    router = RuleBasedRouter()

    q1 = "What is the maximum lethal dose of acetaminophen?"
    assert router.classify_question(q1) == RoutingDecision.EVIDENCE_CRITICAL

    q2 = "What are the absolute contraindications for metformin therapy?"
    assert router.classify_question(q2) == RoutingDecision.EVIDENCE_CRITICAL

    ret_config = router.get_retrieval_config(RoutingDecision.EVIDENCE_CRITICAL)
    assert ret_config["enable_reranker"] is True
    assert ret_config["final_top_k"] >= 10


def test_router_simple_query() -> None:
    """Test brief single-fact inquiries routed to SIMPLE."""
    router = RuleBasedRouter()

    q = "Define hypertension."
    decision = router.classify_question(q)
    assert decision in (RoutingDecision.SIMPLE, RoutingDecision.STANDARD)

    ret_config = router.get_retrieval_config(decision)
    assert ret_config["final_top_k"] <= 5


def test_router_complex_query() -> None:
    """Test comparative inquiries routed to COMPLEX."""
    router = RuleBasedRouter()

    q = "Compare and contrast the efficacy of ACE inhibitors versus ARBs in diabetic nephropathy."
    decision = router.classify_question(q)
    assert decision == RoutingDecision.COMPLEX

    complexity = router.assess_complexity(q)
    assert complexity in (QuestionComplexity.MEDIUM, QuestionComplexity.HIGH)


def test_router_configuration_overrides() -> None:
    """Test router parameters change according to routing decision."""
    router = RuleBasedRouter()

    ret_simple = router.get_retrieval_config(RoutingDecision.SIMPLE)
    assert ret_simple["final_top_k"] == 3
    assert ret_simple["enable_reranker"] is False

    ret_crit = router.get_retrieval_config(RoutingDecision.EVIDENCE_CRITICAL)
    assert ret_crit["final_top_k"] == 10
    assert ret_crit["enable_reranker"] is True

    gen_crit = router.get_generation_config(RoutingDecision.EVIDENCE_CRITICAL)
    assert gen_crit["temperature"] <= 0.1

    ver_crit = router.get_verification_config(RoutingDecision.EVIDENCE_CRITICAL)
    assert ver_crit["required"] is True
    assert ver_crit["strict_mode"] is True


def test_adaptive_router_classification_and_history() -> None:
    """Test AdaptiveRouter classification, history tracking, and policy update."""
    adaptive = AdaptiveRouter()

    q = "What are early symptoms of multiple sclerosis?"
    decision = adaptive.classify_question(q)
    assert decision in RoutingDecision
    assert len(adaptive.history) == 1
    assert adaptive.history[0]["decision"] == decision.value

    # Test update policy
    adaptive.update_policy([{"accuracy": 0.92, "risk": 0.05}])
