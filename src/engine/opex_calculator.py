"""Scale OPEX and Token Economics Calculator.

Calculates token volumes and recurring pass-through SaaS costs at scale:
- 10,000 queries/day
- 100,000 queries/day
- 1,000,000 queries/day (audited scale)
Against $0 marginal token cost on sovereign dedicated hardware.
"""
from typing import Dict
from pydantic import BaseModel


class ScaleTierProjection(BaseModel):
    queries_per_day: int
    daily_tokens: int
    daily_cost_usd: float
    monthly_cost_usd: float
    annual_cost_usd: float
    sovereign_marginal_cost_usd: float = 0.0
    margin_savings_usd: float


class OpexComparison(BaseModel):
    node_name: str
    tokens_per_query: int
    cost_per_query_usd: float
    projections: Dict[str, ScaleTierProjection]


class OpexCalculator:
    """Computes empirical token accounting and enterprise OPEX projections."""

    def __init__(self, cost_per_m_tokens: float = 0.042):
        self.cost_per_m_tokens = cost_per_m_tokens

    def project_scale(self, node_name: str, tokens_per_query: int) -> OpexComparison:
        cost_per_token = self.cost_per_m_tokens / 1_000_000.0
        cost_per_query = tokens_per_query * cost_per_token

        tiers = {
            "10k": 10_000,
            "100k": 100_000,
            "1M": 1_000_000,
        }

        projections = {}
        for tier_name, q_count in tiers.items():
            daily_tokens = q_count * tokens_per_query
            daily_cost = daily_tokens * cost_per_token
            monthly_cost = daily_cost * 30.0
            annual_cost = daily_cost * 365.0

            projections[tier_name] = ScaleTierProjection(
                queries_per_day=q_count,
                daily_tokens=daily_tokens,
                daily_cost_usd=round(daily_cost, 2),
                monthly_cost_usd=round(monthly_cost, 2),
                annual_cost_usd=round(annual_cost, 2),
                sovereign_marginal_cost_usd=0.0,
                margin_savings_usd=round(monthly_cost, 2),
            )

        return OpexComparison(
            node_name=node_name,
            tokens_per_query=tokens_per_query,
            cost_per_query_usd=cost_per_query,
            projections=projections,
        )
