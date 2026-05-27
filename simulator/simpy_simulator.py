from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import simpy


@dataclass
class RequestRecord:
    request_id: int
    arrival_time: float
    prompt_len: int
    generation_len: int
    kv_required: int
    accepted: bool
    admission_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    ttft: float | None = None
    latency: float | None = None
    rejected_time: float | None = None


@dataclass
class ActiveRequest:
    record: RequestRecord
    prefill_done: bool = False
    decoded_tokens: int = 0


def iteration_latency(batch_size: int, tokens: int, phase: str) -> float:
    """Simple workload-dependent latency curve in simulated seconds."""
    if batch_size <= 0:
        return 0.0

    batch_factor = 1.0 + 0.22 * (batch_size - 1) ** 0.65
    if phase == "prefill":
        return 0.004 + 0.00008 * tokens * batch_factor
    return 0.006 + 0.00055 * batch_factor


class ContinuousBatchingSimulator:
    def __init__(
        self,
        *,
        duration: float,
        arrival_rate: float,
        max_batch_size: int,
        kv_budget: int,
        seed: int = 1,
        prompt_mean: float = 256.0,
        generation_mean: float = 128.0,
    ) -> None:
        self.duration = duration
        self.arrival_rate = arrival_rate
        self.max_batch_size = max_batch_size
        self.kv_budget = kv_budget
        self.prompt_mean = prompt_mean
        self.generation_mean = generation_mean

        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)
        self.records: list[RequestRecord] = []
        self.waiting: deque[ActiveRequest] = deque()
        self.active: list[ActiveRequest] = []
        self.reserved_kv = 0
        self.work_available = self.env.event()

    def run(self) -> list[RequestRecord]:
        self.env.process(self._arrival_process())
        self.env.process(self._server_process())
        self.env.run(until=self.duration)
        return self.records

    def _arrival_process(self):
        request_id = 0
        while self.env.now < self.duration:
            interarrival = self.rng.exponential(1.0 / self.arrival_rate)
            yield self.env.timeout(interarrival)
            if self.env.now > self.duration:
                break

            prompt_len = max(1, int(self.rng.lognormal(np.log(self.prompt_mean), 0.55)))
            generation_len = max(
                1, int(self.rng.lognormal(np.log(self.generation_mean), 0.70))
            )
            kv_required = prompt_len + generation_len
            record = RequestRecord(
                request_id=request_id,
                arrival_time=self.env.now,
                prompt_len=prompt_len,
                generation_len=generation_len,
                kv_required=kv_required,
                accepted=False,
            )
            request_id += 1

            if self.reserved_kv + kv_required <= self.kv_budget:
                record.accepted = True
                record.admission_time = self.env.now
                self.reserved_kv += kv_required
                self.waiting.append(ActiveRequest(record=record))
                self._wake_server()
            else:
                record.rejected_time = self.env.now

            self.records.append(record)

    def _server_process(self):
        while True:
            if not self.active and not self.waiting:
                self.work_available = self.env.event()
                yield self.work_available

            self._fill_batch()
            if not self.active:
                continue

            prefill_requests = [req for req in self.active if not req.prefill_done]
            if prefill_requests:
                tokens = sum(req.record.prompt_len for req in prefill_requests)
                yield self.env.timeout(
                    iteration_latency(len(self.active), tokens, "prefill")
                )
                for req in prefill_requests:
                    req.prefill_done = True
                continue

            yield self.env.timeout(iteration_latency(len(self.active), 1, "decode"))
            finished = []
            for req in self.active:
                req.decoded_tokens += 1
                if req.decoded_tokens == 1:
                    req.record.first_token_time = self.env.now
                    req.record.ttft = req.record.first_token_time - req.record.arrival_time
                if req.decoded_tokens >= req.record.generation_len:
                    req.record.finish_time = self.env.now
                    req.record.latency = req.record.finish_time - req.record.arrival_time
                    self.reserved_kv -= req.record.kv_required
                    finished.append(req)

            if finished:
                finished_ids = {id(req) for req in finished}
                self.active = [req for req in self.active if id(req) not in finished_ids]

    def _fill_batch(self) -> None:
        while len(self.active) < self.max_batch_size and self.waiting:
            self.active.append(self.waiting.popleft())

    def _wake_server(self) -> None:
        if not self.work_available.triggered:
            self.work_available.succeed()
