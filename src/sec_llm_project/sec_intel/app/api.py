"""HTTP API for demonstrations (docs/API_PLAN.md).

One worker thread owns the pipeline, because the machine runs one model at a time;
requests only enqueue. A question is submitted, then polled or streamed:

    POST   /v1/questions              {question, ticker?, mode: "fast" | "verified"}
    GET    /v1/questions/{id}         status, stage, remaining seconds, result
    GET    /v1/questions/{id}/events  server-sent events until the answer is done
    DELETE /v1/questions/{id}         cancel a queued question
    GET    /v1/estimate?mode=fast     expected seconds for a new question now
    GET    /v1/health                 index loaded, queue length
    GET    /                          demo page

Fast mode skips the judge (no claim verdicts, no refusal on weak support); verified mode
runs it. Expected times are rolling medians of each stage's measured duration.

    SEC_API_KEY=... sec-intel serve --config config/local_16gb_v6.yaml
"""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import statistics
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 500
MAX_QUEUE = 20
STAGES = {"fast": ("retrieving", "answering"),
          "verified": ("retrieving", "answering", "verifying")}
# Seeds until real questions are measured (EXPERIMENTS.md §16, §20.2): HyDE + search,
# the answer, and the judge (verdicts format ~34 s, claims format ~68 s incl. load).
SEED_SECONDS = {"retrieving": 24.0, "answering": 30.0, "verifying": 34.0}
FAST_WARNING = ("Fast mode: this answer was not checked by the independent judge. "
                "Use mode=verified for claim-by-claim verification.")
_ACTIVE = {"queued", "retrieving", "answering", "verifying"}


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    ticker: str | None = Field(default=None, max_length=10)
    mode: Literal["fast", "verified"] = "fast"


# ---------------------------------------------------------------- estimates
class Estimator:
    """Rolling median duration of each stage, per mode (last 50 questions)."""

    def __init__(self, seed: dict[str, float] | None = None) -> None:
        self.seed = {**SEED_SECONDS, **(seed or {})}
        self._d: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, mode: str, stage: str, seconds: float) -> None:
        with self._lock:
            self._d.setdefault((mode, stage), deque(maxlen=50)).append(seconds)

    def stage(self, mode: str, stage: str) -> float:
        with self._lock:
            d = self._d.get((mode, stage))
            return statistics.median(d) if d else self.seed[stage]

    def total(self, mode: str) -> float:
        return sum(self.stage(mode, s) for s in STAGES[mode])

    def remaining(self, mode: str, stage: str, in_stage_s: float) -> float:
        """Seconds left for a running question in ``stage`` for ``in_stage_s``."""
        stages = STAGES[mode]
        i = stages.index(stage) if stage in stages else 0
        left = max(self.stage(mode, stages[i]) - in_stage_s, 2.0)
        return left + sum(self.stage(mode, s) for s in stages[i + 1:])

    def samples(self, mode: str) -> int:
        with self._lock:
            return min((len(self._d.get((mode, s), ())) for s in STAGES[mode]), default=0)


# --------------------------------------------------------------------- store
class JobStore:
    """Jobs and results in SQLite, so results survive a restart."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, "
                "question TEXT, ticker TEXT, mode TEXT, filters TEXT, resolution TEXT, "
                "status TEXT, started REAL, finished REAL, result TEXT, error TEXT, "
                "estimate_s REAL)")
            # a question that was queued or running when the server stopped will not run
            self._db.execute("UPDATE jobs SET status='failed', error='server restarted' "
                             "WHERE status IN ('queued','retrieving','answering','verifying')")
            self._db.commit()

    def add(self, **row: Any) -> None:
        cols = ", ".join(row)
        with self._lock:
            self._db.execute(f"INSERT INTO jobs ({cols}) VALUES ({', '.join('?' * len(row))})",
                             list(row.values()))
            self._db.commit()

    def update(self, job_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE jobs SET {sets} WHERE id=?", [*fields.values(), job_id])
            self._db.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def queued(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created").fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------------- service
class QAService:
    """Queue + single worker thread around one pipeline."""

    def __init__(self, pipeline: Any, store: JobStore, estimator: Estimator | None = None
                 ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.estimator = estimator or Estimator(_seed_from(pipeline))
        self._q: queue.Queue[str] = queue.Queue()
        self.current: dict[str, Any] | None = None  # {id, mode, stage, stage_start}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qa-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put("")

    # ----------------------------------------------------------- submit
    def submit(self, q: QuestionIn) -> dict[str, Any]:
        if len(self.store.queued()) >= MAX_QUEUE:
            raise HTTPException(429, f"queue full ({MAX_QUEUE} questions); try again later")
        if q.ticker:
            filters: dict[str, Any] | None = {"ticker": q.ticker.upper()}
            resolution = {"ticker": q.ticker.upper(), "how": "given"}
        else:
            filters, res = self.pipeline.resolve_filters(q.question)
            if filters is None:
                raise HTTPException(422, "Could not tell which company the question is "
                                         "about. Name the company or pass 'ticker'.")
            resolution = {"ticker": filters["ticker"], "how": "resolved",
                          "evidence": res.evidence}
        job_id = uuid.uuid4().hex[:12]
        est = self.wait_before(len(self.store.queued())) + self.estimator.total(q.mode)
        self.store.add(id=job_id, created=time.time(), question=q.question,
                       ticker=filters["ticker"], mode=q.mode, filters=json.dumps(filters),
                       resolution=json.dumps(resolution), status="queued", estimate_s=est)
        self._q.put(job_id)
        return self.status(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such question")
        if job["status"] != "queued":
            raise HTTPException(409, f"cannot cancel a question that is {job['status']}")
        self.store.update(job_id, status="cancelled", finished=time.time())
        return self.status(job_id)

    # ----------------------------------------------------------- estimates
    def _current_remaining(self) -> float:
        cur = self.current
        if not cur:
            return 0.0
        return self.estimator.remaining(cur["mode"], cur["stage"],
                                        time.time() - cur["stage_start"])

    def wait_before(self, position: int) -> float:
        """Seconds until a question at queue ``position`` (0 = next) starts."""
        ahead = self.store.queued()[:position]
        return self._current_remaining() + sum(self.estimator.total(j["mode"]) for j in ahead)

    def estimate(self, mode: str) -> dict[str, Any]:
        n = len(self.store.queued())
        wait = self.wait_before(n)
        own = self.estimator.total(mode)
        return {"mode": mode, "estimated_seconds": round(wait + own), "wait_seconds": round(wait),
                "answer_seconds": round(own), "queue_length": n + (1 if self.current else 0),
                "based_on_questions": self.estimator.samples(mode),
                "stages": {s: round(self.estimator.stage(mode, s)) for s in STAGES[mode]}}

    def status(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such question")
        out: dict[str, Any] = {
            "id": job_id, "status": job["status"], "mode": job["mode"],
            "question": job["question"], "ticker": job["ticker"],
            "resolution": json.loads(job["resolution"] or "null"),
            "created": job["created"], "estimated_seconds_at_submit": round(job["estimate_s"] or 0),
            "poll_url": f"/v1/questions/{job_id}", "events_url": f"/v1/questions/{job_id}/events"}
        if job["status"] == "queued":
            ids = [j["id"] for j in self.store.queued()]
            pos = ids.index(job_id) if job_id in ids else 0
            out["queue_position"] = pos + 1
            out["estimated_seconds_remaining"] = round(
                self.wait_before(pos) + self.estimator.total(job["mode"]))
        elif job["status"] in _ACTIVE:
            out["stage"] = job["status"]
            out["estimated_seconds_remaining"] = round(self._current_remaining())
            out["elapsed_seconds"] = round(time.time() - (job["started"] or time.time()))
        if job["result"]:
            out["result"] = json.loads(job["result"])
        if job["error"]:
            out["error"] = job["error"]
        return out

    # ----------------------------------------------------------- worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._q.get()
            if not job_id:
                continue
            job = self.store.get(job_id)
            if job is None or job["status"] != "queued":  # cancelled meanwhile
                continue
            self._run(job)

    def _run(self, job: dict[str, Any]) -> None:
        mode, t0 = job["mode"], time.time()
        timings: dict[str, float] = {}
        self.current = {"id": job["id"], "mode": mode, "stage": "retrieving", "stage_start": t0}
        self.store.update(job["id"], status="retrieving", started=t0)

        def on_stage(stage: str) -> None:
            now, cur = time.time(), self.current
            if cur and cur["stage"] != stage:
                timings[cur["stage"]] = now - cur["stage_start"]
                self.current = {**cur, "stage": stage, "stage_start": now}
                self.store.update(job["id"], status=stage)

        try:
            answer = self.pipeline.ask(job["question"], filters=json.loads(job["filters"]),
                                       verify=mode == "verified", on_stage=on_stage)
            cur = self.current
            if cur:
                timings[cur["stage"]] = time.time() - cur["stage_start"]
            if not answer.abstained or set(timings) >= set(STAGES[mode]):
                for stage, sec in timings.items():  # abstentions skip stages: not recorded
                    self.estimator.record(mode, stage, sec)
            result = serialize_answer(answer, mode, timings, time.time() - t0,
                                      self.pipeline.config)
            self.store.update(job["id"], status="done", finished=time.time(),
                              result=json.dumps(result))
        except Exception as exc:  # one failed question must not stop the worker
            self.store.update(job["id"], status="failed", finished=time.time(),
                              error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            self.current = None


def _seed_from(pipeline: Any) -> dict[str, float]:
    cfg = getattr(pipeline, "config", None)
    fmt = getattr(getattr(cfg, "verification", None), "judge_format", "claims")
    return {"verifying": 34.0 if fmt == "verdicts" else 68.0}


def serialize_answer(answer: Any, mode: str, timings: dict[str, float], total_s: float,
                     config: Any) -> dict[str, Any]:
    """The API's result body: answer, numbered sources, verdicts (verified), timings."""
    extra = answer.extra or {}
    ranked = [r["chunk_id"] for r in extra.get("retrieval", [])]
    by_id = {c.chunk_id: c for c in answer.citations}
    sources = []
    from ..generation.verifier import cited_numbers

    for n in cited_numbers(answer.text):
        cid = ranked[n - 1] if 0 < n <= len(ranked) else None
        c = by_id.get(cid) if cid else None
        if c is not None:
            sources.append({"n": n, "company": c.company, "ticker": c.ticker,
                            "form": c.filing_type, "filing_date": c.filing_date,
                            "section": c.section_title, "item": c.item_number,
                            "url": c.source_url, "excerpt": c.quote})
    out: dict[str, Any] = {
        "answer": answer.text, "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason, "verified": mode == "verified",
        "sources": sources,
        "timings": {k: round(v, 1) for k, v in timings.items()}, "total_seconds": round(total_s, 1),
        "models": {"generator": getattr(config.llm, "model", ""),
                   "judge": getattr(config.verification.judge, "model", "")
                   if mode == "verified" else None}}
    if mode == "fast":
        out["warning"] = FAST_WARNING
    ver = extra.get("verification")
    if mode == "verified" and ver:
        out["groundedness"] = ver.get("groundedness")
        out["claims"] = [{"claim": c["claim"], "verdict": c["verdict"],
                          "source": c.get("citation")}
                         for c in ver.get("claims", []) if not c.get("meta")]
        if extra.get("pruned"):
            out["removed_unsupported"] = extra["pruned"]["removed_sentences"]
    return out


# ----------------------------------------------------------------------- app
def create_app(pipeline: Any = None, *, config_path: str | None = None,
               api_key: str | None = None, db_path: str = "data/api/jobs.db",
               build_pipeline: Callable[[str], Any] | None = None) -> FastAPI:
    """``pipeline`` for tests; otherwise one is built from ``config_path`` at startup.
    With ``api_key`` set, every /v1 request needs ``Authorization: Bearer <key>``."""
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipe = pipeline
        if pipe is None:
            pipe = (build_pipeline or _build_pipeline)(config_path or "config/local_16gb_v6.yaml")
        svc = QAService(pipe, JobStore(db_path))
        svc.start()
        state["svc"] = svc
        yield
        svc.stop()

    app = FastAPI(title="sec-intel", version="1", lifespan=lifespan,
                  docs_url=None if api_key else "/docs", redoc_url=None)

    def auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "missing or wrong API key",
                                headers={"WWW-Authenticate": "Bearer"})

    def svc() -> QAService:
        return state["svc"]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")

    @app.get("/v1/health")
    def health(s: QAService = Depends(svc)) -> dict[str, Any]:
        return {"status": "ok", "queue_length": len(s.store.queued()),
                "busy": s.current is not None,
                "generator": getattr(s.pipeline.config.llm, "model", "")}

    @app.get("/v1/estimate", dependencies=[Depends(auth)])
    def estimate(mode: Literal["fast", "verified"] = Query("fast"),
                 s: QAService = Depends(svc)) -> dict[str, Any]:
        return s.estimate(mode)

    @app.post("/v1/questions", status_code=202, dependencies=[Depends(auth)])
    def submit(q: QuestionIn, s: QAService = Depends(svc)) -> dict[str, Any]:
        return s.submit(q)

    @app.get("/v1/questions/{job_id}", dependencies=[Depends(auth)])
    def get_status(job_id: str, s: QAService = Depends(svc)) -> dict[str, Any]:
        return s.status(job_id)

    @app.delete("/v1/questions/{job_id}", dependencies=[Depends(auth)])
    def cancel(job_id: str, s: QAService = Depends(svc)) -> dict[str, Any]:
        return s.cancel(job_id)

    @app.get("/v1/questions/{job_id}/events", dependencies=[Depends(auth)])
    async def events(job_id: str, s: QAService = Depends(svc)) -> StreamingResponse:
        s.status(job_id)  # 404 before the stream starts

        async def gen():
            last = None
            while True:
                st = s.status(job_id)
                key = (st["status"], st.get("queue_position"))
                if key != last or st["status"] in _ACTIVE:
                    yield f"event: status\ndata: {json.dumps(st)}\n\n"
                    last = key
                if st["status"] not in _ACTIVE:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def _build_pipeline(config_path: str) -> Any:
    from ...utils.env_config import load_environment
    from ..core.config import AppConfig
    from ..pipeline import SECIntelPipeline

    load_environment()
    pipe = SECIntelPipeline(AppConfig.load(config_path))
    pipe.load_index()
    _ = pipe.resolver  # build the company resolver before the first request
    return pipe
