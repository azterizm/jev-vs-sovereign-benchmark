"""Offline mock fixtures for dry-run validation.

Provides realistic response traces mirroring the exact JSON schemas returned by
OpenRouter's Decisions API for typesafe/jev-1.13.
"""
from typing import Any, Dict

MOCK_JEV_RESPONSES: Dict[str, Dict[str, Any]] = {
    # Node 1 Intent Probes
    "n1-comp-01": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n1-01",
        "provider": "TypeSafe",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "company",
                "probabilities": {"company": 0.99, "employment": 0.01, "insolvency": 0.00, "general": 0.00},
                "confidence": 0.99,
            }
        },
        "usage": {"input_tokens": 340, "output_tokens": 28, "cost": 0.00001428},
    },
    "n1-emp-01": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n1-02",
        "provider": "TypeSafe",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "employment",
                "probabilities": {"employment": 0.99, "company": 0.01, "insolvency": 0.00, "general": 0.00},
                "confidence": 0.99,
            }
        },
        "usage": {"input_tokens": 352, "output_tokens": 28, "cost": 0.00001478},
    },
    "n1-ins-01": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n1-03",
        "provider": "TypeSafe",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "insolvency",
                "probabilities": {"insolvency": 0.98, "company": 0.02, "employment": 0.00, "general": 0.00},
                "confidence": 0.98,
            }
        },
        "usage": {"input_tokens": 348, "output_tokens": 28, "cost": 0.00001462},
    },
    # Default fallback for Node 1
    "n1_default": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n1-def",
        "provider": "TypeSafe",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 0.90, "company": 0.05, "employment": 0.03, "insolvency": 0.02},
                "confidence": 0.90,
            }
        },
        "usage": {"input_tokens": 320, "output_tokens": 28, "cost": 0.00001344},
    },
    # Node 2 Pairwise Rerank Mock
    "n2_pairwise": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n2-pair",
        "provider": "TypeSafe",
        "answers": {
            "relevance": {
                "type": "noul",
                "noul": 0.92,
            }
        },
        "usage": {"input_tokens": 485, "output_tokens": 16, "cost": 0.00002037},
    },
    # Node 3 Verification Mocks
    "n3-grounded-era124": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n3-01",
        "provider": "TypeSafe",
        "answers": {
            "verification": {
                "type": "choice",
                "choice": "supports",
                "probabilities": {"supports": 0.98, "contradicts": 0.01, "says_nothing": 0.01},
                "confidence": 0.97,
            }
        },
        "usage": {"input_tokens": 420, "output_tokens": 45, "cost": 0.00001764},
    },
    "n3-contradiction-threshold": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n3-02",
        "provider": "TypeSafe",
        "answers": {
            "verification": {
                "type": "choice",
                "choice": "contradicts",
                "probabilities": {"supports": 0.02, "contradicts": 0.96, "says_nothing": 0.02},
                "confidence": 0.95,
            }
        },
        "usage": {"input_tokens": 428, "output_tokens": 45, "cost": 0.00001798},
    },
    # Adversarial Probe: Jev falls into "confidently wrong" trap (peaked softmax over fictional statute)
    "n3-adversarial-marchwood-01": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n3-adv1",
        "provider": "TypeSafe",
        "answers": {
            "verification": {
                "type": "choice",
                "choice": "supports",
                "probabilities": {"supports": 0.97, "contradicts": 0.01, "says_nothing": 0.02},
                "confidence": 0.96,  # Statistically peaked, but epistemically hallucinated
            }
        },
        "usage": {"input_tokens": 445, "output_tokens": 45, "cost": 0.00001869},
    },
    # Deontic Dilution Probe: Jev treats modal shift ("shall" -> "may") as neutral or supporting
    "n3-deontic-dilution-shall-to-may": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n3-deontic",
        "provider": "TypeSafe",
        "answers": {
            "verification": {
                "type": "choice",
                "choice": "supports",
                "probabilities": {"supports": 0.88, "contradicts": 0.05, "says_nothing": 0.07},
                "confidence": 0.84,  # Fails to penalize modal dilution
            }
        },
        "usage": {"input_tokens": 435, "output_tokens": 45, "cost": 0.00001827},
    },
    "n3_default": {
        "model": "typesafe/jev-1.13-20260917",
        "id": "gen-dec-dryrun-n3-def",
        "provider": "TypeSafe",
        "answers": {
            "verification": {
                "type": "choice",
                "choice": "supports",
                "probabilities": {"supports": 0.85, "contradicts": 0.05, "says_nothing": 0.10},
                "confidence": 0.82,
            }
        },
        "usage": {"input_tokens": 410, "output_tokens": 45, "cost": 0.00001722},
    },
}
