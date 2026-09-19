"""Unit tests for cryptographic audit sealing."""
import json
import pytest
from src.engine.audit_seal import AuditSealer, AuditSeal


def test_seal_generation_and_verification():
    sample_results = {
        "node1": {"p50": 0.08, "tokens": 0},
        "node2": {"p50": 2.4, "tokens": 0},
        "node3": {"p50": 18.2, "tokens": 0},
    }
    sample_dataset = [{"id": "probe-1", "query": "test query"}]

    seal = AuditSealer.generate_seal(
        benchmark_results=sample_results,
        dataset_content=sample_dataset,
    )

    assert seal.seal_id.startswith("SEAL-SOV-JEV-")
    assert len(seal.canonical_seal_hash) == 64
    assert len(seal.benchmark_payload_sha256) == 64
    assert len(seal.dataset_sha256) == 64
    assert seal.compliance_standard == "EU AI Act Article 15 (Accuracy & Deterministic Robustness)"

    # Verify payload matching
    assert seal.verify_payload(sample_results) is True

    # Tampered payload fails
    tampered_results = {
        "node1": {"p50": 999.0, "tokens": 0},
    }
    assert seal.verify_payload(tampered_results) is False
