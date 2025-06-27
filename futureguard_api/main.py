"""
main.py  –  FastAPI entry‑point for FutureGuard

This module exposes a small HTTP API that powers the FutureGuard
web‑application: a financial wellness dashboard that forecasts a user’s
cash balance via Monte‑Carlo simulation.

Endpoints

GET  /ping                                – Lightweight health‑check.
POST /run_risk_analysis/{user_id}         – Runs the heavy ML + Monte‑Carlo pipeline and returns the daily percentiles/best/worst paths.
GET  /account_overview/{user_id}?…        – Returns current balance, recent transactions and UI metadata.
POST /download_risk_analysis/{user_id}    – Creates a ZIP (PNG chart + CSV raw data) so the UI can offer the user an offline download.
"""

from __future__ import annotations

# Standard library 
from pathlib import Path
from datetime import date, timedelta
from typing import Any, Dict, Tuple
from io import BytesIO, StringIO
import zipfile
import time

# Third‑party 
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Local imports 
from risk_pipeline import run_full_analysis  # Our heavy ML + simulation core

# ---------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------

# Root of the project - adjust to your local setup.
BASE_DIR = Path(
    r"C:/Users/loics/OneDrive/Documents/1. BAM/BLOCK 5/Assignment coding/futureguard_api"
)

app = FastAPI(title="FutureGuard API", version="0.5.0")

# Allow JavaScript clients to call us from any origin. In production we may
# want something stricter, but for a student assignment this should be enough.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
#  VERY SMALL IN‑PROCESS CACHE
# ---------------------------------------------------------------------

# We keep the results of expensive simulations in memory for one hour so that
# the UI can re‑poll without hammering the CPU. This is NOT suitable for
# production traffic or multi‑process deployments. it’s intentionally tiny
# and simple for the sake of this assignment.

_CACHE_TTL = 60 * 60  # seconds → 1 hour
# key   = (user_id, "YYYY‑MM‑DD", horizon)
# value = (timestamp_when_cached, result_dict)
_analysis_cache: Dict[Tuple[int, str, int], Tuple[float, dict]] = {}

def _cache_get(user_id: int, start: date, horizon: int) -> dict | None:
    """Return a cached analysis if it is still fresh, otherwise ``None``.

    Parameters
    ----------
    user_id : int
        ID of the user owning the account.
    start : datetime.date
        Simulation start date.
    horizon : int
        Number of days simulated.

    Returns
    -------
    dict | None
        Cached result or ``None`` when not cached/stale.
    """
    key = (user_id, start.isoformat(), horizon)
    hit = _analysis_cache.get(key)
    if not hit:
        return None 
    ts, res = hit
    if time.time() - ts > _CACHE_TTL:
        _analysis_cache.pop(key, None)  
        return None
    return res  

def _cache_set(user_id: int, start: date, horizon: int, res: dict) -> None:
    """Insert/overwrite a cache entry (thin wrapper around the dict)."""
    key = (user_id, start.isoformat(), horizon)
    _analysis_cache[key] = (time.time(), res)

# ---------------------------------------------------------------------
#  SMALL UTILITIES
# ---------------------------------------------------------------------

def _round_2(x: float) -> float:
    """Round to two decimals & cast away numpy scalar types for JSON serialisation."""
    return float(round(x, 2))

# --------------------------------------------------------------------
#  Helpers for the ZIP download endpoint
# --------------------------------------------------------------------

def _csv_from_result(res: dict[str, Any]) -> bytes:
    """Convert the simulation result into a UTF‑8 CSV.

    The columns match the keys used by the UI so that front‑end code can reuse
    them as‑is. The function returns raw bytes so that the caller can insert
    them straight into the in‑memory ZIP archive without touching disk.
    """
    df = pd.DataFrame({
        "date":   res["dates"],
        "median": res["median"],
        "p5":     res["p5"],
        "p95":    res["p95"],
        "best":   res["best"],
        "worst":  res["worst"],
    })
    buff = StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode()

def _plot_png_from_result(res: dict[str, Any]) -> bytes:
    """Create a simple Matplotlib fan‑chart (PNG) from the simulation output."""
    dates = pd.to_datetime(res["dates"])

    # Create a small figure.
    fig, ax = plt.subplots(figsize=(10, 4))

    # Shade the inter‑quartile band first so that the median/best/worst lines remain fully visible on top.
    ax.fill_between(dates, res["p5"], res["p95"], alpha=.15, label="5–95 %")

    # Overlay the deterministic lines.
    ax.plot(dates, res["median"], lw=2, label="Median", color="orange")
    ax.plot(dates, res["best"],   ls="--", label="Best",   color="green")
    ax.plot(dates, res["worst"],  ls="--", label="Worst",  color="red")

    # Axes labelling & layout tweaks.
    ax.set_ylabel("Balance (€)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()

    # Serialise to bytes, close the figure to free the memory and hand back the PNG payload.
    png = BytesIO()
    fig.savefig(png, format="png", dpi=160)
    plt.close(fig)
    png.seek(0)
    return png.read()

def _zip_bytes(*, png: bytes, csv: bytes) -> bytes:
    """Pack the provided artefacts into an in‑memory ZIP and return bytes."""
    buff = BytesIO()
    with zipfile.ZipFile(buff, "w") as zf:
        zf.writestr("balance_forecast.png", png)
        zf.writestr("forecast_data.csv",    csv)
    buff.seek(0)
    return buff.read()

# ---------------------------------------------------------------------
#  ENDPOINT IMPLEMENTATIONS
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
#  /ping – Health‑check
# ---------------------------------------------------------------------

@app.get("/ping")
def ping() -> dict[str, bool]:
    """A single‑field JSON so cloud load‑balancers know we’re alive."""
    return {"ok": True}

# ---------------------------------------------------------------------
#  /run_risk_analysis/{user_id} – Heavy ML + simulation
# ---------------------------------------------------------------------

@app.post("/run_risk_analysis/{user_id}")
def run_risk_analysis(
    user_id: int,
    start: date = Query(..., description="Forecast start (YYYY-MM-DD)"),
    horizon: int = Query(92, ge=1, le=365, description="Days to simulate"),
):
    """Run the expensive Monte‑Carlo pipeline (10k paths) for *one* user."""

    # 1. Short‑circuit if we have a fresh cache entry.
    cached = _cache_get(user_id, start, horizon)
    if cached is not None:
        return cached

    # 2. Otherwise defer to the heavy backend.
    try:
        res = run_full_analysis(
            user_id=user_id,
            base_dir=BASE_DIR,
            start_date=start,
            horizon=horizon,
            n_sim=10_000,
            progress_cb=None,  # CLI prints only – REST callers ignore progress
        )
        # 3. Populate the cache so subsequent calls are instant.
        _cache_set(user_id, start, horizon, res)
        return res

    # 4 Map common exceptions to meaningful HTTP status codes.
    except FileNotFoundError as exc:
        raise HTTPException(400, f"Missing file: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error: {exc}") from exc

# ---------------------------------------------------------------------
#  /account_overview/{user_id} – Balance & recent transactions
# ---------------------------------------------------------------------
@app.get("/account_overview/{user_id}")
def account_overview(
    user_id: int,
    start: date | None = Query(
        None,
        description=(
            "Forecast start (YYYY-MM-DD). "
            "If omitted, first-tx date + 90 days."
        ),
    ),
    horizon: int = Query(
        92, ge=1, le=365,
        description="Forecast horizon in days (echoed back to UI).",
    ),
) -> dict[str, Any]:
    """Return header info, current balance & recent transactions for *one* user."""

    try:
        # 1. Read & normalise the synthetic test CSV (see README) 
        csv = BASE_DIR / "synthetic_transactions_enriched.csv"
        df = pd.read_csv(csv, parse_dates=["date"])
        df.columns = [c.lower().strip() for c in df.columns]  # snake‑case cols

        # Sanity‑check that the must‑have columns exist.
        must_have = {"user_id", "date", "amount"}
        if not must_have.issubset(df.columns):
            raise HTTPException(
                500,
                f"CSV missing required cols: {', '.join(must_have - set(df.columns))}",
            )

        # Harmonise the counter‑party column
        if "counterparty" not in df.columns:
            for alt in ("merchant", "description"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "counterparty"})
                    break
        df["counterparty"] = df.get("counterparty", pd.Series(["–"] * len(df)))
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        # 2. Isolate *one* user & sort chronologically 
        u = df[df["user_id"] == user_id].copy()
        if u.empty:
            raise HTTPException(404, f"user_id={user_id} not found in CSV")
        u.sort_values("date", inplace=True)

        # 3. Default <start> = first transaction + 90 days if not provided 
        if start is None:
            start = (u["date"].min() + timedelta(days=90)).date()

        # 4. Opening balance = all tx strictly before <start> 
        bal = u[u["date"] < pd.Timestamp(start)]["amount"].sum()

        # 5. Header info (name & IBAN)
        first = u.iloc[0]
        name = first.get("name") or first.get("account_name") or f"User {user_id}"
        iban = first.get("iban") or first.get("IBAN") or "–"

        # 6. Recent transactions window (latest ≤ <start>) 
        tx = (
            u[u["date"] < pd.Timestamp(start)]
            .loc[:, ["date", "counterparty", "amount"]]
            .sort_values("date", ascending=False)
            .head(250)
        )
        tx["date"] = tx["date"].dt.strftime("%Y-%m-%d")
        tx["amount"] = tx["amount"].apply(_round_2)

        # Sidebar widgets displayed by the front‑end; hard‑coded for demo.
        widgets = ["Transfer", "Request", "Look Ahead", "Settings"]

        # Shape the JSON payload expected by the React UI.
        return {
            "user":    {"id": user_id, "name": str(name), "iban": str(iban)},
            "balance": _round_2(bal),
            "start":   start.isoformat(),
            "horizon": horizon,
            "widgets": widgets,
            "tx":      tx.to_dict(orient="records"),
        }

    # Map I/O and other errors to clean HTTP status codes.
    except FileNotFoundError:
        raise HTTPException(400, f"CSV not found: {csv}") from None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error: {exc}") from exc

# ---------------------------------------------------------------------
#  /download_risk_analysis/{user_id} – ZIP (PNG + CSV)
# ---------------------------------------------------------------------
@app.post("/download_risk_analysis/{user_id}")
def download_risk_analysis(
    user_id: int,
    start: date = Query(..., description="Forecast start (YYYY-MM-DD)"),
    horizon: int = Query(92, ge=1, le=365, description="Days to simulate"),
):
    """Return an in‑memory ZIP so the front‑end can offer a download button."""

    # 1. Try the in‑memory cache first to avoid re‑running the heavy pipeline.
    res = _cache_get(user_id, start, horizon)
    if res is None:
        # 2. Fallback: run the simulation and populate the cache for next time
        try:
            res = run_full_analysis(
                user_id=user_id,
                base_dir=BASE_DIR,
                start_date=start,
                horizon=horizon,
                n_sim=10_000,
                progress_cb=None,
            )
            _cache_set(user_id, start, horizon, res)
        except FileNotFoundError as exc:
            raise HTTPException(400, f"Missing file: {exc}") from exc
        except Exception as exc:
            raise HTTPException(500, f"Unexpected error: {exc}") from exc

    # 3. Create the artefacts and stream the ZIP (no temp files on disk).
    zip_bytes = _zip_bytes(
        png=_plot_png_from_result(res),
        csv=_csv_from_result(res),
    )

    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="risk_analysis_{user_id}.zip"'
        },
    )

