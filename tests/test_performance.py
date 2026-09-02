from __future__ import annotations

import json
import resource
import statistics
import time
from typing import Any

import pytest

from test_dialects import _load_pipeline


BENCHMARK_TEXT = (
    "Your verification code is 654321. "
    "Use it to confirm your secure login within 5 minutes."
)
REQUEST_COUNT = 1_000
MAX_AVERAGE_LATENCY_MS = 15.0
MAX_RSS_MB = 150.0


@pytest.fixture(scope="module")
def pipeline() -> Any:
    return _load_pipeline()


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024.0 if usage < 10_000_000 else 1_048_576.0)


def test_1000_requests_latency_memory_and_stability(pipeline: Any) -> None:
    for _ in range(10):
        result = json.loads(pipeline.predict_json(BENCHMARK_TEXT))
        assert isinstance(result, dict)

    before_rss = _rss_mb()
    latencies_ms: list[float] = []
    for _ in range(REQUEST_COUNT):
        started = time.perf_counter()
        result = json.loads(pipeline.predict_json(BENCHMARK_TEXT))
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        assert isinstance(result, dict)

    after_rss = _rss_mb()
    average_latency_ms = statistics.fmean(latencies_ms)
    assert average_latency_ms <= MAX_AVERAGE_LATENCY_MS
    assert after_rss <= MAX_RSS_MB
    assert after_rss - before_rss <= 5.0