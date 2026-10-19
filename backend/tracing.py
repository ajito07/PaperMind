"""Per-answer pipeline trace: stage payloads + timings, persisted per message."""
import json
import time
from contextlib import contextmanager


class Trace:
    def __init__(self):
        self.stages: list[tuple[str, dict, int]] = []

    @contextmanager
    def stage(self, name: str):
        payload: dict = {}
        t0 = time.perf_counter()
        try:
            yield payload
        finally:
            self.stages.append((name, payload, int((time.perf_counter() - t0) * 1000)))

    def persist(self, conn, message_id: int):
        conn.executemany(
            "INSERT INTO traces(message_id, stage, payload, duration_ms) VALUES (?,?,?,?)",
            [(message_id, name, json.dumps(payload, default=str), ms)
             for name, payload, ms in self.stages],
        )
