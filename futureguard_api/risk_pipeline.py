"""
risk_pipeline.py – compact, self‑contained version
(only the parts that changed are commented with  ### NEW / FIX ###)

This module trains three lightweight Transformer models to create daily
probabilistic forecasts of income and expenses for a single user.

High‑level steps
================
1.  Read historical transactions from a CSV file.
2.  Perform feature engineering (calendar, holidays, categorical flags).
3.  Construct sliding windows to build encoder / decoder tensors.
4.  Train three separate models:
      * `clf` – binary classifier for “salary paid?” flag.
      * `reg` – quantile regressor (p10 / p50 / p90) for income amount.
      * `exp` – quantile regressor (p10 / p50 / p90) for expense amount.
5.  Generate forecasts for the requested horizon and save them to disk.

The code is intentionally minimal: a single file with no external
dependencies beyond NumPy, Pandas, Matplotlib, TensorFlow 2.x and
python‑dateutil.
"""

from __future__ import annotations  # allow postponed evaluation of type hints (Python ≥3.7)

# ────────────────────────────────────────────────────────────────────────────────
# Standard‑library imports
# ────────────────────────────────────────────────────────────────────────────────
import json           # for serialising metadata (not used directly below but kept for parity)
import random         # reproducible pseudo‑randomness when training
import time           # optional timestamps for progress callbacks
from datetime import date, timedelta  # convenient date arithmetic
from pathlib import Path             # portable file‑system paths
from typing import Callable, Dict, Tuple  # static type hints

# ────────────────────────────────────────────────────────────────────────────────
# Third‑party imports
# ────────────────────────────────────────────────────────────────────────────────
import numpy as np                    # vectorised maths
import pandas as pd                   # tabular data handling
import matplotlib.pyplot as plt       # plotting (only required if you add charts)
import matplotlib.dates as mdates     # helper to format dates on plots
from dateutil.easter import easter    # compute Easter Sunday for Dutch holidays

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, MultiHeadAttention,
    LayerNormalization, Dropout, Add,
)
from tensorflow.keras.models import Model

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def _notify(step: str, pct: int, cb: Callable[[str, int], None] | None):
    """Invoke an optional callback to report *step* progress.

    Parameters
    ----------
    step : str
        Arbitrary label, e.g. ``"train-clf"`` or ``"forecast"``.
    pct : int
        Percentage complete (0–100). Only a hint – the function is free
        to call back at any granularity.
    cb : callable | None
        A user‑supplied progress callback. When *None*, the function is a
        no‑op so we can sprinkle calls without if‑checks elsewhere.
    """
    if cb is not None:
        cb(step, pct)


# ---------------------------------------------------------------------
# SECTION 1 TRANSFORMER MODEL BUILDER
# ---------------------------------------------------------------------

# The architecture is deliberately modest (small *d_model*, few layers)
# so that it trains quickly on CPU/GPU

def _build_transformer(
    L: int, H: int, enc_dim: int, dec_dim: int,
    *,                                  # L, H = Encoder/decoder sequence lengths; enc_dim, dec_dim = feature dimensions for encoder and decoder
    d_model: int = 64,                  # Size of the shared embedding space
    n_heads: int = 4,                   # Number of attention heads
    ff_dim: int = 256,                  # Hidden size of the position‑wise feed‑forward network
    do_rate: float = 0.1,               # Drop‑out rate applied after attention/FFN blocks
    n_layers: int = 2,                  # Repeated encoder–decoder blocks
    last_activation: str = "linear",    # Activation for the final Dense layer (e.g. "sigmoid" for binary classification)
    output_units: int = 1,              # Dimensionality of the model output
) -> Model:
    # Inputs 
    enc_in = Input((L, enc_dim))  # shape: (batch, L, enc_dim)
    dec_in = Input((H, dec_dim))  # shape: (batch, H, dec_dim)

    # Positional encoding helper
    def _pos_enc(T: int) -> tf.Tensor:
        pos = np.arange(T)[:, None]                 # (T, 1)
        i   = np.arange(d_model)[None, :]           # (1, d_model)
        angle = pos / (10000 ** (2 * (i // 2) / d_model))
        pe = np.zeros((T, d_model), np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])        # even indices -> sin
        pe[:, 1::2] = np.cos(angle[:, 1::2])        # odd  indices - > cos
        return tf.constant(pe)                      # (T, d_model)

    # Embed + add positional information
    x_enc = Dense(d_model)(enc_in) + _pos_enc(L)
    x_dec = Dense(d_model)(dec_in) + _pos_enc(H)

    # Stack of n_layers encoder–decoder blocks
    for _ in range(n_layers):
        # Encoder self‑attention block 
        attn = MultiHeadAttention(n_heads, d_model, dropout=do_rate)(x_enc, x_enc)
        x_enc = LayerNormalization(epsilon=1e-6)(Add()([x_enc, Dropout(do_rate)(attn)]))

        ff = Dense(ff_dim, "relu")(x_enc)
        ff = Dense(d_model)(ff)
        x_enc = LayerNormalization(epsilon=1e-6)(Add()([x_enc, Dropout(do_rate)(ff)]))

        # Decoder self‑attention 
        attn = MultiHeadAttention(n_heads, d_model, dropout=do_rate)(
            x_dec, x_dec, use_causal_mask=True
        )
        x_dec = LayerNormalization(epsilon=1e-6)(Add()([x_dec, Dropout(do_rate)(attn)]))

        # Encoder–decoder cross‑attention 
        cross = MultiHeadAttention(n_heads, d_model, dropout=do_rate)(x_dec, x_enc, x_enc)
        x_dec = LayerNormalization(epsilon=1e-6)(Add()([x_dec, Dropout(do_rate)(cross)]))

        # Decoder feed‑forward 
        ff = Dense(ff_dim, "relu")(x_dec)
        ff = Dense(d_model)(ff)
        x_dec = LayerNormalization(epsilon=1e-6)(Add()([x_dec, Dropout(do_rate)(ff)]))

    # Final projection (per‑time‑step outputs) 
    out = Dense(output_units, activation=last_activation)(x_dec)
    return Model([enc_in, dec_in], out)


# ────────────────────────────────────────────────────────────────────────────────
# Main entry‑point: fit models & forecast
# ────────────────────────────────────────────────────────────────────────────────

def _fit_forecast_models(
    user_id: int, # Only rows matching this ID are used for training/forecasting.
    hist_csv: Path, # CSV containing historical daily `income`, `expenses`, `date`, `user_id` columns.
    forecast_start: pd.Timestamp, # Inclusive date range to predict.
    forecast_end:   pd.Timestamp, # Inclusive date range to predict.
    *,
    seed: int = 42, # Random seed for reproducibility across NumPy, Python *random* and TensorFlow.
    progress_cb: Callable[[str, int], None] | None = None, # Optional callback to receive progress updates (see `_notify`).
) -> Tuple[pd.DataFrame, Path]:
    _notify("loading-data", 5, progress_cb)  # let the UI know we started

    # Ensure determinism 
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Load & filter data 
    df = pd.read_csv(hist_csv, parse_dates=["date"])
    user_df = (
        df[df["user_id"] == user_id]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 2. Feature engineering
    user_df["day_of_week"]       = user_df["date"].dt.weekday  # 0=Mon
    user_df["month"]             = user_df["date"].dt.month
    user_df["day_of_month"]      = user_df["date"].dt.day
    user_df["is_first_of_month"] = (user_df["day_of_month"] == 1).astype(int)

    # Dutch (NL) public holidays helper 
    def _nl_holidays(y0, y1):
        """Return a *set* of public holiday dates between *y0* and *y1* (incl.)."""
        hol = set()
        for yr in range(y0, y1 + 1):
            hol.add(date(yr, 1, 1))                         # New Year's Day
            e = easter(yr)                                  # Easter Sunday
            hol |= {e - timedelta(2),                       # Good Friday
                    e + timedelta(1),                       # Easter Monday
                    e + timedelta(39),                      # Ascension Day
                    e + timedelta(50)}                      # Whit Monday
            # King's Day – shift if falls on Sunday
            k = date(yr, 4, 26 if date(yr, 4, 27).weekday() == 6 else 27)
            hol.add(k)
            if yr == 2025:
                hol.add(date(yr, 5, 5))                     # Liberation Day (quinquennial)
            hol |= {date(yr, 12, 25), date(yr, 12, 26)}     # Christmas
        return hol

    user_df["holiday"] = user_df["date"].dt.date.isin(
        _nl_holidays(user_df["date"].dt.year.min(), 2025)  # extend until horizon
    ).astype(int)

    # 3. Prepare sliding‑window dataset 
    L = 365                                         # encoder length: one full year of history
    H = (forecast_end - forecast_start).days + 1    # decoder length

    train_df = user_df[user_df["date"] < forecast_start].copy()

    # Normalise target series 
    income_series  = train_df["income"].values
    expense_series = train_df["expenses"].values

    income_max       = income_series.max() or 1.0          # prevent /0
    expense_min_abs  = abs(expense_series.min()) or 1.0

    income_scaled  = income_series  / income_max
    expense_scaled = expense_series / (-expense_min_abs)   

    # Categorical one‑hot features 
    n = len(train_df)
    dow = np.zeros((n, 7),  np.float32)
    mon = np.zeros((n, 12), np.float32)
    dow[np.arange(n), train_df["day_of_week"]] = 1
    mon[np.arange(n), train_df["month"] - 1]   = 1

    first_of_month = train_df["is_first_of_month"].astype(np.float32).values.reshape(-1, 1)
    holiday_flag   = train_df["holiday"].astype(np.float32).values.reshape(-1, 1)

    # Split windows into training / validation by end date
    tr_idx, val_idx = [], []
    for s in range(n - L - H + 1):
        end_date = train_df.iloc[s + L + H - 1]["date"]
        (tr_idx if end_date <= pd.Timestamp("2024-12-31") else val_idx).append(s)

    tr_idx, val_idx = np.array(tr_idx), np.array(val_idx)

    # Helper to build (Xe, Xd, y) tensors from starting indices 
    def _make_xy(idxs: np.ndarray):
        Xe, Xd, y_flag, y_inc, y_exp = [], [], [], [], []
        for i in idxs:
            # Skip windows that would run past the end 
            if i + L + H > n:
                continue

            hist = slice(i, i + L)       # encoder slice
            fut  = slice(i + L, i + L + H)  # decoder slice

            # Encoder: history features 
            Xe.append(np.hstack([
                income_scaled[hist].reshape(L, 1),
                expense_scaled[hist].reshape(L, 1),
                dow[hist], mon[hist], first_of_month[hist], holiday_flag[hist]
            ]))

            # Decoder: only calendar / categorical flags 
            Xd.append(np.hstack([
                dow[fut], mon[fut], first_of_month[fut], holiday_flag[fut]
            ]))

            # Targets 
            inc_fut = income_scaled[fut]
            y_flag.append((inc_fut > 0).astype(np.float32))  # salary indicator
            y_inc.append(inc_fut)                            # raw scaled amounts
            y_exp.append(expense_scaled[fut])

        # Use *stack* to guarantee identical shapes  ### NEW / FIX ###
        return (
            np.stack(Xe).astype(np.float32),
            np.stack(Xd).astype(np.float32),
            np.stack(y_flag).astype(np.float32),
            np.stack(y_inc ).astype(np.float32),
            np.stack(y_exp ).astype(np.float32),
        )

    Xe_tr, Xd_tr, y_flag_tr, y_inc_tr, y_exp_tr = _make_xy(tr_idx)
    Xe_val, Xd_val, y_flag_val, y_inc_val, y_exp_val = (
        _make_xy(val_idx) if len(val_idx) else (None,) * 5
    )

    enc_dim, dec_dim = Xe_tr.shape[2], Xd_tr.shape[2]

    # For quantile losses 
    quantiles = tf.constant([0.1, 0.5, 0.9], tf.float32)
    Q = 3  # number of quantiles

    # Generic compile/fit wrapper 
    def _compile_fit(model, loss_fn, y_tr, y_val, tag):  
        _notify(f"train-{tag}", 0, progress_cb)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
            loss=loss_fn,
            metrics=[] if tag != "clf" else ["accuracy"],
        )

        callbacks = []
        if y_val is not None:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            )

        model.fit(
            [Xe_tr, Xd_tr],
            y_tr,
            validation_data=([Xe_val, Xd_val], y_val) if y_val is not None else None,
            epochs=50,
            batch_size=32,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )
        _notify(f"train-{tag}", 100, progress_cb)

    # 4. Train models
    # 4a. Binary classifier – salary yes/no
    clf = _build_transformer(L, H, enc_dim, dec_dim, last_activation="sigmoid", output_units=1)
    _compile_fit(
        clf,
        "binary_crossentropy",
        y_flag_tr[..., None],
        y_flag_val[..., None] if Xe_val is not None else None,
        "clf",
    )

    # 4b. Quantile regressor – income amount 
    def masked_pinball(y_t, y_p):
        # Only backpropagate when an income is expected (mask == 1)
        mask = tf.cast(y_t[..., 0] > 0, tf.float32)
        e = y_t - y_p
        q = quantiles[None, None, :]
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e) * mask[..., None])

    reg = _build_transformer(L, H, enc_dim, dec_dim, output_units=Q)
    _compile_fit(
        reg,
        masked_pinball,
        np.repeat(y_inc_tr[..., None], Q, 2),
        np.repeat(y_inc_val[..., None], Q, 2) if Xe_val is not None else None,
        "reg",
    )

    # 4c. Quantile regressor – expenses
    def pinball(y_t, y_p):
        e = y_t - y_p
        q = quantiles[None, None, :]
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

    exp = _build_transformer(L, H, enc_dim, dec_dim, output_units=Q)
    _compile_fit(
        exp,
        pinball,
        np.repeat(y_exp_tr[..., None], Q, 2),
        np.repeat(y_exp_val[..., None], Q, 2) if Xe_val is not None else None,
        "exp",
    )

    # 5. Forecast
    _notify("forecast", 15, progress_cb)

    # Helper to build a single‑sample encoder input from the latest history 
    def _enc_block(series_scaled):
        return np.hstack([
            series_scaled[-L:].reshape(L, 1),           # income (scaled)
            expense_scaled[-L:].reshape(L, 1),          # expenses (scaled)
            dow[-L:], mon[-L:], first_of_month[-L:], holiday_flag[-L:],
        ])[None, ...]  # add batch dimension

    enc_in = _enc_block(income_scaled)

    # Decoder (future) categorical features
    fd = pd.date_range(forecast_start, forecast_end, freq="D")

    dec_dow = np.zeros((H, 7))
    dec_dow[np.arange(H), fd.weekday] = 1

    dec_mon = np.zeros((H, 12))
    dec_mon[np.arange(H), fd.month - 1] = 1

    dec_first = (fd.day == 1).astype(np.float32).reshape(-1, 1)
    dec_hol = np.array([d.date() in _nl_holidays(2022, 2025) for d in fd], np.float32).reshape(-1, 1)

    dec_in = np.hstack([dec_dow, dec_mon, dec_first, dec_hol])[None, ...]

    # Run models
    prob_salary  = clf.predict([enc_in, dec_in], verbose=0)[0].squeeze()
    inc_amount_s = reg.predict([enc_in, dec_in], verbose=0)[0]  # scaled
    exp_amount_s = exp.predict([enc_in, dec_in], verbose=0)[0]  # scaled

    # Post‑process
    inc_amount_s *= (prob_salary > 0.5)[:, None]                    # zero out improbable salary days
    inc_amount = inc_amount_s * income_max
    exp_amount = np.minimum(exp_amount_s * (-expense_min_abs), 0)   # keep as negative values

    pred_df = pd.DataFrame({
        "date":           fd.date,
        "income_prob":    prob_salary,
        "income_p10":     inc_amount[:, 0],
        "income_median":  inc_amount[:, 1],
        "income_p90":     inc_amount[:, 2],
        "expense_p10":    exp_amount[:, 0],
        "expense_median": exp_amount[:, 1],
        "expense_p90":    exp_amount[:, 2],
    })
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # 6. Persist to disk
    out_path = hist_csv.parent / "final_income_expense_predictions.csv"
    pred_df.to_csv(out_path, index=False)

    _notify("forecast", 40, progress_cb)
    return pred_df, out_path


# ---------------------------------------------------------------------
# SECTION 2 - MONTE-CARLO + RISK METRICS
# ---------------------------------------------------------------------
def _run_mc_simulation(
    pred_df: pd.DataFrame, # Output of the forecasting models with daily percentile predictions for income and expenses 
    hist_df: pd.DataFrame, # Historical daily net-cash-flow for the same user 
    user_id: int, # ID of the user being analysed (only used for logging/debugging)
    *,
    forecast_start: pd.Timestamp, # First day of the forward-looking period (the day _after_ the history ends)
    num_sim: int = 10_000, # Number of Monte-Carlo paths to generate
    progress_cb: Callable[[str, int], None] | None = None, # Hook for UI progress-bar updates
) -> Dict:

    # --------------------------------------------------------------- 
    # 2-a - PREP HISTORICAL BALANCE                                   
    # --------------------------------------------------------------- 
    _notify("mc", 40, progress_cb)  # 40 % done in the overall pipeline

    # Keep only dates strictly BEFORE the forecast window
    hist_df = (
        hist_df[hist_df["date"] < forecast_start]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Turn daily net-cash-flows into a running balance
    hist_df["balance"] = hist_df["net_cash_flow"].cumsum()
    initial_balance = hist_df.iloc[-1]["balance"] if len(hist_df) else 0.0

    # Last 90 historical days -> blue line on the chart
    cutoff       = forecast_start - pd.Timedelta(days=90)
    hist_last3   = hist_df[hist_df["date"] >= cutoff]
    hist_dates   = hist_last3["date"].dt.strftime("%Y-%m-%d").tolist()
    hist_vals    = hist_last3["balance"].round(2).tolist()

    # --------------------------------------------------------------- 
    # 2-b - MONTE-CARLO SIMULATION                                    
    # --------------------------------------------------------------- 
    num_days = len(pred_df)                      # simulation horizon (= forecast_df rows)
    dates    = pred_df["date"]                   # convenience alias
    sim      = np.zeros((num_sim, num_days), dtype=np.float32)   # (paths × days)

    for s in range(num_sim):
        bal = initial_balance                    # start each path at the last real balance
        for i, row in pred_df.iterrows():
            # Income and expense are sampled independently from their P10–P90 range
            inc = np.random.uniform(row["income_p10"],  row["income_p90"])
            exp = np.random.uniform(min(row["expense_p10"], row["expense_p90"]),
                                    max(row["expense_p10"], row["expense_p90"]))
            bal += inc + exp
            sim[s, i] = bal

        # Progress callback roughly every 10 %
        if (s + 1) % max(1, num_sim // 10) == 0:
            _notify("mc", 40 + int((s + 1) / num_sim * 40), progress_cb)

    # --------------------------------------------------------------- 
    # 2-c - PERCENTILE ENVELOPE FOR THE FAN CHART                     
    # --------------------------------------------------------------- 
    median_path = np.percentile(sim, 50, axis=0)   # median trajectory
    p5_path     = np.percentile(sim,  5, axis=0)   # 5th-percentile (pessimistic band)
    p95_path    = np.percentile(sim, 95, axis=0)   # 95th-percentile (optimistic band)

    final_balances = sim[:, -1]                    # final balance of each path
    best_path      = sim[final_balances.argmax()]  # single best path
    worst_path     = sim[final_balances.argmin()]  # single worst path

    # --------------------------------------------------------------- 
    # 2-d - RISK METRICS (VaR / CVaR)                                 
    # --------------------------------------------------------------- 
    returns_92d = final_balances - initial_balance          # absolute € change at horizon
    VaR_95  = np.percentile(returns_92d,  5)                # loss threshold exceeded 5 % of the time
    VaR_99  = np.percentile(returns_92d,  1)                # …1 % of the time
    CVaR_95 = returns_92d[returns_92d <= VaR_95].mean()     # average of the worst 5 %
    CVaR_99 = returns_92d[returns_92d <= VaR_99].mean()     # average of the worst 1 %

    # --------------------------------------------------------------- 
    # 2-e - RULE-BASED NATURAL-LANGUAGE INSIGHTS                      
    # --------------------------------------------------------------- 
    def r100(x: float) -> int:                      # helper → round to the nearest €100 for readability
        return int(np.round(x, -2))

    prob_od     = (sim < 0).any(1).mean()           # probability of hitting negative balance
    mean_end    = final_balances.mean()
    median_end  = np.median(final_balances)
    median_low  = np.median(sim.min(axis=1))        # typical intra-path low
    buf_needed  = max(0, -VaR_95)                   # buffer to make 95 % VaR ≥ 0
    monthly_gap = -(mean_end - initial_balance) / (num_days / 30)  # projected deficit / month

    best_95  = np.percentile(final_balances, 95)   # optimistic bracket
    worst_95 = np.percentile(final_balances,  5)   # pessimistic bracket

    # Narrative intro paragraph
    intro = (
        f"We predicted your income and expenses for the next {num_days} days "
        f"and simulated {num_sim:,} potential paths. "
        f"On average the balance ends around €{r100(median_end):,}. "
        f"In 95 % of cases it lands between €{r100(best_95):,} "
        f"and €{r100(worst_95):,}. "
        f"With a confidence of 95 %, your worst-case loss (Value at Risk) is roughly €{abs(r100(VaR_95)):,}."
    )

    # Headline depends on starting point and overdraft likelihood
    if initial_balance < 0:
        headline = "⚠️ Already overdrawn; risk of deeper overdraft ahead."
    elif prob_od < 0.20 and mean_end >= 0:
        headline = "Your outlook is healthy; overdraft risk is low."
    elif prob_od < 0.60:
        headline = "Caution: you could dip below zero in some scenarios."
    else:
        headline = "⚠️ High overdraft risk: most paths fall below €0."

    # Supporting sentence about where the median ends (or bottoms)
    support = (
        f"Median projection {'bottoms at −€' if initial_balance < 0 else 'ends at €'}"
        f"{r100(median_low if initial_balance < 0 else median_end):,}."
    )
    # Add comment on structural gap if large
    if abs(monthly_gap) > 500:
        if monthly_gap > 0:
            support += f" You’re expected to spend about €{r100(monthly_gap):,} more than you earn each month."
        else:
            support += f" You’re expected to earn about €{r100(-monthly_gap):,} more than you spend each month."

    # Actionable bullet-point recommendations (max 3)
    bullets: list[str] = []
    if initial_balance < 0:
        bullets.append(f"Top-up €{r100(-initial_balance):,} now to reach €0.")
        if buf_needed:
            bullets.append(f"Add another €{r100(buf_needed):,} to stay safe (95 %).")
    else:
        if buf_needed:
            bullets.append(f"Save €{r100(buf_needed):,} to avoid overdraft in 95 % of cases.")

    if monthly_gap > 500 and len(bullets) < 3:
        pct = int(min(30, max(5, monthly_gap / (abs(monthly_gap) + 1) * 10)))
        bullets.append(f"Cut monthly spending {pct}% to slow the deficit.")
    if prob_od > 0.60 and len(bullets) < 3:
        bullets.append("Enable low-balance alerts before hitting €0.")
    if initial_balance > 0 and prob_od < 0.20 and len(bullets) < 3:
        bullets.append("Consider moving surplus to a savings or investment plan.")
    if len(bullets) < 3:
        bullets.append("Review subscriptions and cancel unused ones.")
    bullets = bullets[:3]                                       # ensure only top 3

    _notify("mc", 100, progress_cb)                             # mark simulation step as complete

    # --------------------------------------------------------------- 
    # 2-f - PACKAGE RESULTS FOR FastAPI                               
    # --------------------------------------------------------------- 
    return {
        # Blue historical line data
        "hist_dates"   : hist_dates,
        "hist_balances": hist_vals,

        # Forecast fan-chart series
        "dates" : dates.dt.strftime("%Y-%m-%d").tolist(),
        "median": median_path.tolist(),
        "p5"    : p5_path.tolist(),
        "p95"   : p95_path.tolist(),
        "best"  : best_path.tolist(),
        "worst" : worst_path.tolist(),

        # Narrative text blocks
        "insights"       : [intro, headline, support],
        "recommendations": bullets,

        # Extra metrics (optional for the UI)
        "initial_balance": float(initial_balance),
        "VaR_95"         : float(abs(VaR_95)),
        "VaR_99"         : float(abs(VaR_99)),
        "CVaR_95"        : float(abs(CVaR_95)),
        "CVaR_99"        : float(abs(CVaR_99)),
    }


# ---------------------------------------------------------------------
# SECTION 3 - ORCHESTRATOR
# ---------------------------------------------------------------------
def run_full_analysis(
    user_id: int,
    base_dir: str | Path,
    *,
    start_date: date,
    horizon: int,
    n_sim: int = 10_000,
    progress_cb: Callable[[str, int], None] | None = None,
) -> Dict:
    """
    High-level workflow that glues together:
      1. Loading historical data
      2. Fitting/predicting future cash-flows
      3. Running the Monte-Carlo & risk metrics
      4. Returning a JSON payload ready for the API/UI layer.
    """
    base_dir = Path(base_dir)
    hist_csv = base_dir / "synthetic_transactions_daily_enriched.csv"
    hist_df  = pd.read_csv(hist_csv, parse_dates=["date"])

    # Define forecast window [fs, fe]
    fs = pd.Timestamp(start_date)
    fe = fs + pd.Timedelta(days=horizon - 1)

    # 1. Train + infer daily income/expense percentiles
    pred_df, _ = _fit_forecast_models(
        user_id, hist_csv, fs, fe, progress_cb=progress_cb
    )

    # 2️. Feed forecasts + history into the risk engine
    result = _run_mc_simulation(
        pred_df,
        hist_df[hist_df["user_id"] == user_id][["date", "net_cash_flow"]],
        user_id,
        forecast_start=fs,
        num_sim=n_sim,
        progress_cb=progress_cb,
    )

    _notify("done", 100, progress_cb)   # full pipeline finished
    return result

# ---------------------------------------------------------------------
# SECTION 4 - PUBLIC ENTRY POINT (imported by FastAPI)
# ---------------------------------------------------------------------
def run_risk_pipeline(
    user_id: int = 10,
    progress_cb: Callable[[str, int], None] | None = None,
) -> Dict:
    """
    This thin façade is what the FastAPI route calls.
    It simply passes through to `run_full_analysis` with sensible defaults.
    """
    return run_full_analysis(
        user_id=user_id,
        base_dir=Path(__file__).parent,   # directory where the CSV lived
        n_sim=10_000,                     # default number of Monte-Carlo paths
        progress_cb=progress_cb,
    )

# ---------------------------------------------------------------------
# CLI HELPER (allows running `python risk.py` from the shell)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys, time

    ap = argparse.ArgumentParser(
        description="Run FutureGuard risk-analysis pipeline from the shell"
    )
    ap.add_argument("--user_id", type=int, default=10, help="User ID to analyse")
    ap.add_argument(
        "--base_dir",
        type=str,
        default=r"C:/Users/loics/OneDrive/Documents/1. BAM/BLOCK 5/Assignment coding/futureguard_api",
        help="Folder that holds the CSVs",
    )
    ap.add_argument("--n_sim", type=int, default=10_000, help="Number of MC paths")
    args = ap.parse_args()

    # Console progress-bar -> simple visual feedback
    def _console(step: str, pct: int) -> None:
        bar = ("█" * (pct // 5)).ljust(20)
        print(f"[{bar}] {pct:3d}%  {step}")

    t0 = time.perf_counter()
    print(f"🚀  Starting pipeline for user {args.user_id} …")
    res = run_full_analysis(
        user_id=args.user_id,
        base_dir=args.base_dir,
        n_sim=args.n_sim,
        progress_cb=_console,
    )
    dur = time.perf_counter() - t0
    print(f"\n✅  Done in {dur/60:,.1f} minutes")
    print(json.dumps(res, indent=2))
    sys.exit(0)
