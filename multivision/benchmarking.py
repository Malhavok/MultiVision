"""Small thread-safe timing counters shared by the running-service benchmark."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class _TimingAggregate:
    sample_count: int = 0
    total_seconds: float = 0.0
    maximum_seconds: float = 0.0
    cpu_seconds: float = 0.0
    stall_count: int = 0


class BenchmarkMetrics:
    """Collect process-local timings without making them part of runtime state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timings: dict[str, _TimingAggregate] = {}
        self._counters: dict[str, int] = {}

    def record_timing(
        self,
        component: str,
        elapsed_seconds: float,
        cpu_seconds: float,
        stalled: bool = False,
    ) -> None:
        if not isinstance(component, str) or len(component) == 0:
            raise ValueError('component must be a non-empty string')
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or not isinstance(cpu_seconds, (int, float))
            or isinstance(cpu_seconds, bool)
            or not math.isfinite(elapsed_seconds)
            or not math.isfinite(cpu_seconds)
            or elapsed_seconds < 0
            or cpu_seconds < 0
        ):
            raise ValueError('timings must be finite and non-negative')
        with self._lock:
            aggregate = self._timings.setdefault(component, _TimingAggregate())
            aggregate.sample_count += 1
            aggregate.total_seconds += elapsed_seconds
            aggregate.maximum_seconds = max(
                aggregate.maximum_seconds,
                elapsed_seconds,
            )
            aggregate.cpu_seconds += cpu_seconds
            if stalled:
                aggregate.stall_count += 1

    def increment(self, counter: str, amount: int = 1) -> None:
        if not isinstance(counter, str) or len(counter) == 0:
            raise ValueError('counter must be a non-empty string')
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError('counter amount must be a non-negative integer')
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + amount

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings = {
                component: {
                    'sample_count': aggregate.sample_count,
                    'total_seconds': aggregate.total_seconds,
                    'maximum_seconds': aggregate.maximum_seconds,
                    'mean_seconds': (
                        aggregate.total_seconds / aggregate.sample_count
                        if aggregate.sample_count > 0
                        else 0.0
                    ),
                    'cpu_seconds': aggregate.cpu_seconds,
                    'stall_count': aggregate.stall_count,
                }
                for component, aggregate in sorted(self._timings.items())
            }
            return {
                'counters': dict(sorted(self._counters.items())),
                'timings': timings,
            }

    def reset(self) -> None:
        with self._lock:
            self._timings.clear()
            self._counters.clear()


@contextmanager
def measure_timing(
    metrics: BenchmarkMetrics,
    component: str,
    stall_threshold_seconds: float | None = None,
) -> Iterator[None]:
    """Record wall and thread CPU time around one runtime operation."""
    started_seconds = time.perf_counter()
    started_cpu_seconds = time.thread_time()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - started_seconds
        cpu_seconds = max(0.0, time.thread_time() - started_cpu_seconds)
        stalled = (
            stall_threshold_seconds is not None
            and elapsed_seconds > stall_threshold_seconds
        )
        metrics.record_timing(
            component,
            elapsed_seconds,
            cpu_seconds,
            stalled,
        )
