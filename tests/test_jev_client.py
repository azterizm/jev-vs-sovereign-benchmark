"""Unit tests for OpenRouter Jev Decisions Client."""
import pytest
from src.clients.jev_client import (
    JevClient,
    JevQuestion,
    JevDecisionResponse,
    JevUsage,
)


def test_question_serialization():
    q = JevQuestion(
        type="choice",
        instructions="Classify intent",
        criteria={"a": "desc a", "b": "desc b"},
    )
    dumped = q.model_dump()
    assert dumped["type"] == "choice"
    assert dumped["instructions"] == "Classify intent"
    assert dumped["criteria"]["a"] == "desc a"


def test_response_parsing():
    raw_response = {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-12345",
        "provider": "TypeSafe",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "employment",
                "probabilities": {"employment": 0.98, "company": 0.02},
                "confidence": 0.98,
            }
        },
        "usage": {
            "input_tokens": 350,
            "output_tokens": 25,
            "cost": 0.000015,
        },
    }

    parsed = JevDecisionResponse(**raw_response)
    assert parsed.model == "typesafe/jev-1.13-20260917"
    assert parsed.id == "gen-dec-12345"
    assert parsed.answers["intent"].choice == "employment"
    assert parsed.answers["intent"].confidence == 0.98
    assert parsed.usage.input_tokens == 350
    assert parsed.usage.cost == 0.000015


def test_client_missing_key_raises():
    client = JevClient(api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        client.decide(state="query text", questions={})


def test_fetch_upstream_latencies_without_key():
    client = JevClient(api_key=None)
    assert client.fetch_upstream_latencies(["gen-123"]) == {}
    assert client.fetch_upstream_latencies([]) == {}


def test_response_with_upstream_latency():
    resp = JevDecisionResponse(
        model="typesafe/jev-1.13",
        id="gen-test-1",
        latency_ms=480.0,
        upstream_latency_ms=395.0,
    )
    assert resp.latency_ms == 480.0
    assert resp.upstream_latency_ms == 395.0
