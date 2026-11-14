"""HTTP API (sec_intel/app/api.py) with a fake pipeline: queue, modes, auth, estimates."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from sec_llm_project.sec_intel.app.api import Estimator, create_app
from sec_llm_project.sec_intel.core.types import Answer, Citation

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakePipeline:
    def __init__(self, gate: threading.Event | None = None):
        self.config = SimpleNamespace(
            llm=SimpleNamespace(model="gen-model"),
            verification=SimpleNamespace(judge=SimpleNamespace(model="judge-model"),
                                         judge_format="verdicts"))
        self.gate = gate
        self.calls = []

    def resolve_filters(self, question):
        if "Waters" in question:
            return {"ticker": "WAT"}, SimpleNamespace(evidence={"WAT": "Waters"})
        return None, SimpleNamespace(evidence={})

    def ask(self, question, *, filters=None, verify=True, on_stage=None):
        self.calls.append((question, filters, verify))
        on_stage("retrieving")
        if self.gate is not None:
            self.gate.wait(5)
        on_stage("answering")
        extra = {"retrieval": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]}
        if verify:
            on_stage("verifying")
            extra["verification"] = {"groundedness": 1.0, "claims": [
                {"claim": "Waters maintains cyber insurance", "verdict": "SUPPORTED",
                 "citation": 2, "meta": False}]}
        cit = Citation(chunk_id="c2", ticker="WAT", company="Waters Corp", filing_type="10-K",
                       filing_date="2025-02-25", section_title="Risk Factors", item_number="1A",
                       source_url="https://www.sec.gov/x", score=0.9, quote="cyber insurance")
        return Answer(question=question, text="Waters maintains cyber insurance [2].",
                      citations=[cit], extra=extra)


def _client(tmp_path, pipeline, key=KEY):
    return TestClient(create_app(pipeline, api_key=key, db_path=str(tmp_path / "jobs.db")))


def _wait(c, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/v1/questions/{job_id}", headers=AUTH).json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_fast_mode_is_default_and_skips_the_judge(tmp_path):
    pipe = FakePipeline()
    with _client(tmp_path, pipe) as c:
        r = c.post("/v1/questions", json={"question": "What does Waters say about cyber?"},
                   headers=AUTH)
        assert r.status_code == 202
        sub = r.json()
        assert sub["mode"] == "fast" and sub["ticker"] == "WAT"
        assert sub["estimated_seconds_at_submit"] > 0
        st = _wait(c, sub["id"])
    res = st["result"]
    assert pipe.calls[0][2] is False  # verify=False
    assert res["verified"] is False and "warning" in res and "claims" not in res
    assert res["sources"][0]["n"] == 2 and res["sources"][0]["company"] == "Waters Corp"
    assert set(res["timings"]) == {"retrieving", "answering"}


def test_verified_mode_returns_claim_verdicts(tmp_path):
    with _client(tmp_path, FakePipeline()) as c:
        sub = c.post("/v1/questions", headers=AUTH,
                     json={"question": "Cyber insurance?", "ticker": "wat",
                           "mode": "verified"}).json()
        res = _wait(c, sub["id"])["result"]
    assert sub["resolution"] == {"ticker": "WAT", "how": "given"}
    assert res["verified"] and res["claims"][0]["verdict"] == "SUPPORTED"
    assert res["models"]["judge"] == "judge-model"


def test_auth_and_validation(tmp_path):
    with _client(tmp_path, FakePipeline()) as c:
        assert c.post("/v1/questions", json={"question": "Waters?"}).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert c.get("/v1/estimate", headers=bad).status_code == 401
        assert c.get("/v1/health").status_code == 200  # no key needed
        assert c.get("/").status_code == 200
        r = c.post("/v1/questions", json={"question": "x" * 501}, headers=AUTH)
        assert r.status_code == 422
        r = c.post("/v1/questions", json={"question": "Which company is this?"}, headers=AUTH)
        assert r.status_code == 422 and "company" in r.json()["detail"]
        assert c.get("/v1/questions/nope", headers=AUTH).status_code == 404


def test_queue_position_estimate_and_cancel(tmp_path):
    gate = threading.Event()
    with _client(tmp_path, FakePipeline(gate)) as c:
        first = c.post("/v1/questions", json={"question": "Waters one"}, headers=AUTH).json()
        time.sleep(0.1)  # the worker picks up the first and blocks on the gate
        second = c.post("/v1/questions", json={"question": "Waters two", "mode": "verified"},
                        headers=AUTH).json()
        st = c.get(f"/v1/questions/{second['id']}", headers=AUTH).json()
        assert st["status"] == "queued" and st["queue_position"] == 1
        est = c.get("/v1/estimate?mode=fast", headers=AUTH).json()
        assert est["queue_length"] == 2 and est["wait_seconds"] > 0
        assert c.delete(f"/v1/questions/{second['id']}", headers=AUTH).json()["status"] == "cancelled"
        assert c.delete(f"/v1/questions/{first['id']}", headers=AUTH).status_code == 409
        gate.set()
        assert _wait(c, first["id"])["status"] == "done"


def test_estimator_uses_medians_after_measurements():
    e = Estimator({"verifying": 34.0})
    assert e.total("fast") == 24.0 + 30.0
    for sec in (10.0, 12.0, 50.0):
        e.record("fast", "answering", sec)
    assert e.stage("fast", "answering") == 12.0
    assert e.remaining("fast", "retrieving", 20.0) == pytest.approx(4.0 + 12.0)
    assert e.remaining("verified", "verifying", 100.0) == 2.0  # overrun: floor


def test_excerpt_is_the_matching_sentence_and_sources_are_sorted():
    from sec_llm_project.sec_intel.app.api import serialize_answer

    cit = lambda cid: Citation(chunk_id=cid, ticker="WAT", company="Waters", filing_type="10-K",  # noqa: E731
                               filing_date="2025-02-25", section_title="Risk", item_number="1A",
                               source_url="", score=0.9, quote="tection modules, as well")
    ans = Answer(question="q", text="Waters keeps cyber insurance [2]. Sales rose [1].",
                 citations=[cit("c2"), cit("c1")],
                 extra={"retrieval": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]})
    texts = {"c1": "Net sales rose 5% in 2024. Other text.",
             "c2": "Training covers protection modules. The Company maintains cyber insurance, "
                   "which may not be sufficient."}
    cfg = SimpleNamespace(llm=SimpleNamespace(model="g"),
                          verification=SimpleNamespace(judge=SimpleNamespace(model="j")))
    out = serialize_answer(ans, "fast", {}, 1.0, cfg, chunk_text=texts.get)
    assert [s["n"] for s in out["sources"]] == [1, 2]
    assert out["sources"][1]["excerpt"].startswith("The Company maintains cyber insurance")
    assert out["sources"][0]["excerpt"] == "Net sales rose 5% in 2024."
    texts["c1"] = "les rose 5% in 2024."
    cut = serialize_answer(ans, "fast", {}, 1.0, cfg, chunk_text=texts.get)
    assert cut["sources"][0]["excerpt"] == "… rose 5% in 2024."
    no_text = serialize_answer(ans, "fast", {}, 1.0, cfg)
    assert no_text["sources"][0]["excerpt"] == "tection modules, as well"


def test_estimates_survive_a_restart(tmp_path):
    with _client(tmp_path, FakePipeline()) as c:
        _wait(c, c.post("/v1/questions", json={"question": "Waters one"}, headers=AUTH).json()["id"])
    with _client(tmp_path, FakePipeline()) as c:  # same jobs.db
        est = c.get("/v1/estimate?mode=fast", headers=AUTH).json()
    assert est["based_on_questions"] == 1 and est["estimated_seconds"] < 10
