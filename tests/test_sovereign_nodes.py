"""Unit tests for Sovereign Pipeline Nodes."""
import pytest
import torch
from src.clients.sovereign_models import (
    SovereignIntentRouter,
    SovereignColBERTReranker,
    SovereignNLIAuditor,
)


@pytest.fixture(scope="module")
def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def test_sovereign_intent_and_coordinate_extraction(device):
    router = SovereignIntentRouter(device=device)

    # Test query with Companies Act 2006 s.382
    q1 = "What was the maximum turnover for a company to qualify as small under section 382 of the Companies Act 2006?"
    res1 = router.route(q1)

    assert res1.intent == "company"
    assert any("Companies Act 2006 s.382" in coord for coord in res1.statutory_coordinates)
    assert res1.latency_ms < 20.0  # Sub-millisecond to low millisecond CPU/GPU forward pass

    # Test query with Employment Rights Act 1996 s.124
    q2 = "Statutory compensation caps for unfair dismissal under section 124 of the Employment Rights Act 1996."
    res2 = router.route(q2)
    assert res2.intent == "employment"
    assert any("Employment Rights Act 1996 s.124" in coord for coord in res2.statutory_coordinates)


def test_sovereign_colbert_maxsim_and_span_attribution(device):
    reranker = SovereignColBERTReranker(device=device)
    query = "What turnover threshold qualifies a company as small under section 382?"
    doc = (
        "Companies Act 2006 Section 382 provides the statutory qualification requirements for small companies. "
        "The qualifying conditions are met by a company in a year in which it satisfies two or more of the following requirements: "
        "turnover does not exceed 10.2 million pounds and balance sheet total does not exceed 5.1 million pounds and "
        "the average number of employees does not exceed 50."
    )

    res = reranker.compute_late_interaction_maxsim(query, doc, candidate_id="test-ca382")
    assert res.maxsim_score > 0.0
    assert res.span_attribution is not None
    assert len(res.span_attribution.split()) <= 50
    assert "10.2 million pounds" in res.span_attribution or "Section 382" in res.span_attribution


def test_sovereign_nli_grounded_entailment(device):
    auditor = SovereignNLIAuditor(device=device)
    premise = "Under section 124 of the Employment Rights Act 1996, the compensatory award is capped at £68,400."
    hypothesis = "ERA 1996 s.124 imposes a statutory cap of £68,400 on compensatory awards."

    res = auditor.audit(premise, hypothesis)
    assert res.verdict == "ENTAILMENT"
    assert res.confidence > 0.85
    assert res.epistemic_status == "GROUNDED"


def test_sovereign_nli_epistemic_abstention_gate(device):
    auditor = SovereignNLIAuditor(device=device)
    premise = "According to the Marchwood Commercial Arbitration Order 2022, all commercial disputes require conciliation."
    hypothesis = "The Marchwood Commercial Arbitration Order 2022 mandates pre-action conciliation."

    # Must cleanly abstain rather than hallucinate or score ungrounded text
    res = auditor.audit(premise, hypothesis)
    assert res.verdict == "ABSTAIN"
    assert res.epistemic_status == "UNGROUNDED_INVENTED_INSTRUMENT"
    assert res.confidence == 1.0


def test_sovereign_nli_deontic_dilution_enforcement(device):
    auditor = SovereignNLIAuditor(device=device)
    premise = "The directors of every company shall prepare accounts for the company for each of its financial years under CA 2006."
    hypothesis = "Directors of a company may at their discretion choose whether to prepare annual accounts."

    res = auditor.audit(premise, hypothesis)
    # Dilution of statutory "shall" to "may" must be enforced as contradiction
    assert res.verdict == "CONTRADICTION"
    assert res.epistemic_status == "DEONTIC_VIOLATION"
