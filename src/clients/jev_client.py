"""OpenRouter Jev Decisions API Client.

Implements typed requests targeting:
POST https://openrouter.ai/api/alpha/decisions
Model: typesafe/jev-1.13
"""
import time
import httpx
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class JevQuestion(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: Union[Dict[str, str], List[str]]


class JevAnswer(BaseModel):
    type: str
    noul: Optional[float] = None
    choice: Optional[str] = None
    score: Optional[Union[float, int]] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class JevUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class JevDecisionResponse(BaseModel):
    model: str
    id: str = ""
    provider: str = "TypeSafe"
    answers: Dict[str, JevAnswer] = Field(default_factory=dict)
    usage: JevUsage = Field(default_factory=JevUsage)
    latency_ms: float = 0.0
    upstream_latency_ms: Optional[float] = None
    status_code: int = 200
    is_mock: bool = False


class JevClient:
    """Synchronous & resilient client for OpenRouter's Decisions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "typesafe/jev-1.13",
        timeout: float = 30.0,
        site_url: str = "https://memonsystems.com",
        site_name: str = "Sovereign Legal AI Benchmark",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.site_url = site_url
        self.site_name = site_name

        self._headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        self._client = httpx.Client(timeout=self.timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def decide(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[JevQuestion, Dict[str, Any]]],
    ) -> JevDecisionResponse:
        """Sends a Decisions API request to OpenRouter for typesafe/jev-1.13."""
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Provide an API key or use --dry-run."
            )

        # Convert questions to raw dict
        formatted_questions = {}
        for q_id, q_val in questions.items():
            if isinstance(q_val, JevQuestion):
                formatted_questions[q_id] = q_val.model_dump()
            elif isinstance(q_val, dict):
                formatted_questions[q_id] = q_val
            else:
                raise ValueError(f"Invalid question structure for {q_id}: {type(q_val)}")

        payload = {
            "model": self.model,
            "state": state,
            "questions": formatted_questions,
        }

        t0 = time.perf_counter()
        resp = self._client.post(self.base_url, headers=self._headers, json=payload)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if resp.status_code != 200:
            error_detail = resp.text
            raise RuntimeError(
                f"OpenRouter Decisions API failed with HTTP {resp.status_code}: {error_detail}"
            )

        data = resp.json()
        model_name = data.get("model", self.model)
        req_id = data.get("id", "")
        provider = data.get("provider", "TypeSafe")
        raw_usage = data.get("usage", {})
        usage = JevUsage(
            input_tokens=raw_usage.get("input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
            cost=float(raw_usage.get("cost", 0.0)),
        )

        answers: Dict[str, JevAnswer] = {}
        for ans_key, ans_dict in data.get("answers", {}).items():
            answers[ans_key] = JevAnswer(**ans_dict)

        return JevDecisionResponse(
            model=model_name,
            id=req_id,
            provider=provider,
            answers=answers,
            usage=usage,
            latency_ms=latency_ms,
            status_code=resp.status_code,
            is_mock=False,
        )

    def fetch_upstream_latencies(self, generation_ids: List[str]) -> Dict[str, float]:
        """Queries OpenRouter generation endpoint to retrieve upstream provider latency (excluding client WAN)."""
        if not self.api_key or not generation_ids:
            return {}

        results: Dict[str, float] = {}
        import asyncio

        async def _fetch_batch():
            headers = {"Authorization": f"Bearer {self.api_key}"}
            batch_size = 50
            async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
                for i in range(0, len(generation_ids), batch_size):
                    chunk = generation_ids[i : i + batch_size]
                    tasks = [
                        client.get(f"https://openrouter.ai/api/v1/generation?id={gid}")
                        for gid in chunk
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    for gid, resp in zip(chunk, responses):
                        if isinstance(resp, httpx.Response) and resp.status_code == 200:
                            gdata = resp.json().get("data", {})
                            presp = gdata.get("provider_responses", [])
                            if presp and presp[0].get("latency") is not None:
                                results[gid] = float(presp[0]["latency"])

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, _fetch_batch()).result()
            else:
                asyncio.run(_fetch_batch())
        except Exception:
            pass

        return results

