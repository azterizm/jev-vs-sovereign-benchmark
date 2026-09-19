"""Dataset package for the legal benchmark battery."""
from .legal_battery import (
    NODE_1_PROBES,
    NODE_2_PROBES,
    NODE_3_PROBES,
    Node1Probe,
    Node2Probe,
    Node3Probe,
    CandidatePassage,
)
from .fixtures import MOCK_JEV_RESPONSES

__all__ = [
    "NODE_1_PROBES",
    "NODE_2_PROBES",
    "NODE_3_PROBES",
    "Node1Probe",
    "Node2Probe",
    "Node3Probe",
    "CandidatePassage",
    "MOCK_JEV_RESPONSES",
]
