"""
jobs.py – tiny in-memory job queue for long-running risk-analysis tasks.
Suitable for local testing; swap for Redis/RQ/Celery in production.
"""

from __future__ import annotations
import uuid, threading, time, traceback
from typing import Callable, Dict

JobStatus = dict  # {"progress": int, "step": str, "done": bool, "result": dict|None, "error": str|None}

_jobs: Dict[str, JobStatus] = {}

def _run_in_thread(fn: Callable, job_id: str, *args, **kw):
    """Worker that executes `fn`, catching errors and updating status."""
    j = _jobs[job_id]
    try:
        def _cb(step: str, pct: int):
            j.update(step=step, progress=pct)

        result = fn(*args, progress_cb=_cb, **kw)
        j.update(progress=100, step="done", done=True, result=result)
    except Exception as e:
        j.update(done=True, error=str(e))
        traceback.print_exc()

def enqueue(fn: Callable, *args, **kw) -> str:
    """Return a `job_id` immediately and start work in a new thread."""
    job_id = uuid.uuid4().hex
    _jobs[job_id] = dict(progress=0, step="queued", done=False,
                         result=None, error=None, started=time.time())
    t = threading.Thread(target=_run_in_thread, args=(fn, job_id, *args), kwargs=kw, daemon=True)
    t.start()
    return job_id

def get(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)
