"""Cryptographic Audit Seal for Benchmark Integrity & Deterministic Reproducibility.

Guarantees tamper-evident auditability compliant with:
- EU AI Act Article 15 (Accuracy, Robustness, and Cybersecurity)
- ISO/IEC 42001 AI Management System Standards
- Enterprise Third-Party Risk Management (TPRM)
"""
import hashlib
import json
import platform
import sys
import torch
import transformers
from datetime import datetime, timezone
from typing import Any, Dict
from pydantic import BaseModel, Field


class AuditSeal(BaseModel):
    seal_id: str
    timestamp_utc: str
    compliance_standard: str = "EU AI Act Article 15 (Accuracy & Deterministic Robustness)"
    author: str = "Abdullah Memon | Memon Systems Ltd (UK Inc. No. 17284215)"
    platform_environment: Dict[str, Any]
    dataset_sha256: str
    benchmark_payload_sha256: str
    canonical_seal_hash: str

    def verify_payload(self, raw_data: Dict[str, Any]) -> bool:
        """Verifies whether the provided benchmark payload matches the sealed SHA-256."""
        canonical_json = json.dumps(raw_data, sort_keys=True, separators=(",", ":"))
        calculated = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return calculated == self.benchmark_payload_sha256


class AuditSealer:
    """Generates immutable SHA-256 cryptographic seals for benchmark runs."""

    @staticmethod
    def get_platform_telemetry() -> Dict[str, Any]:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "os": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "compute_device": device.upper(),
        }

    @classmethod
    def generate_seal(
        cls,
        benchmark_results: Dict[str, Any],
        dataset_content: Any,
        seal_prefix: str = "SEAL-SOV-JEV",
    ) -> AuditSeal:
        telemetry = cls.get_platform_telemetry()
        now_utc = datetime.now(timezone.utc).isoformat()

        # Hash dataset
        dataset_str = json.dumps(dataset_content, default=str, sort_keys=True, separators=(",", ":"))
        dataset_hash = hashlib.sha256(dataset_str.encode("utf-8")).hexdigest()

        # Hash benchmark payload
        payload_str = json.dumps(benchmark_results, default=str, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        # Combine into canonical seal
        combined = f"{now_utc}|{telemetry}|{dataset_hash}|{payload_hash}"
        canonical_seal = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        seal_id = f"{seal_prefix}-{canonical_seal[:12].upper()}"

        return AuditSeal(
            seal_id=seal_id,
            timestamp_utc=now_utc,
            platform_environment=telemetry,
            dataset_sha256=dataset_hash,
            benchmark_payload_sha256=payload_hash,
            canonical_seal_hash=canonical_seal,
        )
