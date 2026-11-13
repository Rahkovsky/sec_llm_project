"""ModelScheduler: concurrent questions share few model switches, same calls each."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sec_llm_project.sec_intel.llm.providers import ModelScheduler

# One question's call sequence: HyDE + answer (generator), verify + qualifier check (judge),
# sometimes a repair (generator) and its re-verify (judge).
PLAIN = ["gen", "gen", "judge", "judge"]
REPAIR = [*PLAIN, "gen", "judge"]


def _run(n_questions: int, workers: int) -> tuple[int, list[list[str]]]:
    sched = ModelScheduler()
    loaded: list[str] = []
    lock = threading.Lock()
    seen: list[list[str]] = []

    def question(i: int) -> list[str]:
        calls = []
        with sched.worker():
            for model in REPAIR if i % 3 == 0 else PLAIN:
                time.sleep(0.001)  # CPU work (retrieval, parsing) between calls
                with sched.turn(model):
                    with lock:
                        if not loaded or loaded[-1] != model:
                            loaded.append(model)
                    calls.append(model)
        return calls

    with ThreadPoolExecutor(workers) as pool:
        seen = list(pool.map(question, range(n_questions)))
    assert sched.switches == len(loaded)
    return sched.switches, seen


def test_scheduler_keeps_each_questions_calls_and_cuts_switches():
    serial, calls1 = _run(24, 1)
    batched, calls8 = _run(24, 8)
    assert calls1 == calls8  # every question makes the same calls in the same order
    assert serial == sum(4 if i % 3 == 0 else 2 for i in range(24))  # incl. the first load
    assert batched <= serial / 3
