"""
Project: Model-Based Backtesting Deployment Script for BTC-USDT-SWAP

Overview  
This script deploys trained machine learning models on historical BTC-USDT-SWAP order book and trade data and generates raw backtesting outputs for later performance evaluation.

The script does NOT need to compute Sharpe ratio, Calmar ratio, drawdown, or other summary performance metrics now.  
Its job is only to replay the market in time order, apply the trained models, simulate quote placement / fills / exits, and write detailed per-decision or per-trade outputs so that performance analysis can be done later in a separate script.

The system uses:
- two models to predict optimal quote offset (one for bid side, one for ask side)
- two models to predict whether to execute the trade (one for bid side, one for ask side)

The design should be modular so different models can be plugged in later.

--------------------------------------------------
1. Objective

Deploy trained models on historical order book and trade data and generate a detailed backtesting result dataset containing all raw fields needed for later evaluation.

This script should:
- replay historical market data in time order
- compute features at decision timestamps
- apply trained models
- simulate fills and exits
- write detailed outputs

This script should NOT:
- compute Sharpe ratio
- compute Calmar ratio
- compute drawdown summary
- compute final strategy analytics

Those metrics will be calculated later from this script’s output.

--------------------------------------------------
2. Data Sources

Instrument:
- BTC-USDT-SWAP

Order book data:
- directory: /Users/luke/Desktop/project_intern/data/data_ob
- format: .data JSON lines
- fields include: ts, action, bids, asks

Trade data:
- directory: /Users/luke/Desktop/project_intern/data/data_trade
- format: CSV
- fields include: created_time, price, size, side

Trained models:
/Users/luke/Desktop/project_intern/code/xxxxxxxxx.json

--------------------------------------------------
3. Time Flow Design

The script must simulate a continuous market replay.

It should:
- replay all order book events in chronological order
- maintain in-memory bid and ask books
- maintain rolling trade history
- maintain rolling mid-price history

Important:
- decisions are NOT made on every order book event
- decisions are made only on fixed sampling timestamps
- sampling grid should match the training pipeline, e.g. every 1000 ms

So the logic is:
- replay all order book events continuously
- use them to keep current market state updated
- trigger model inference only at decision timestamps on the fixed grid

--------------------------------------------------
4. Feature Construction

At each decision timestamp t, compute features using only data available at or before t.

Required features:
- OBI5
- OBI25
- OBI400
- NTR_10s
- mid_std_2s
- spread_bps
- hour_of_day
- trade_flow_10s
- trade_count_10s
- cumulativeVolume_5bps

Feature timing rules:
- trade-based windows use [t - 10s, t], details depend on step1.py logic
- mid-price rolling window uses [t - 2s, t] (not from grid-sampled mids), details depend on step1.py logic
- order book state is the latest reconstructed valid book at time t

No future data may be used in feature construction.

If feature windows are incomplete or market state is invalid, the timestamp may be skipped or marked invalid in output.

--------------------------------------------------
5. Model Inference

At each valid decision timestamp t:

Input:
- feature vector X_t

Models:
1. Offset model:
   - predicts quote offset

2. Execution model:
   - predicts whether to place the quote

Decision logic:
- if execution model says do not execute → record that decision and no order is placed
- if execution model says execute → place quote using predicted offset

The modular design allows us to rewrite/change the model quickly no matter it is regressor or classifier

--------------------------------------------------
6. Quote Construction

Let:
- mid_t = (best_bid + best_ask) / 2

Bid quote:
- bid_quote = mid_t * (1 - offset_bps / 10000)

Ask quote:
- ask_quote = mid_t * (1 + offset_bps / 10000)

--------------------------------------------------
7. Fill Simulation

For an executed quote at timestamp t:

Quote active window:
- (t, t + 1s]

Bid fill conditions:
- seller-initiated trades
- trade price ≤ bid_quote
- cumulative qualifying trade size ≥ 1

Ask fill conditions:
- buyer-initiated trades
- trade price ≥ ask_quote
- cumulative qualifying trade size ≥ 1

Rules:
- fill price = quote price (same as in step1.py)
- no partial fills
- first timestamp where cumulative size threshold is reached is fill_ts
- if no fill occurs, mark no-fill outcome

--------------------------------------------------
8. Exit Logic

If a quote is filled at fill_ts:

- exit_target_ts = fill_ts + 1s
- scan forward through order book events
- take the first order book event with timestamp ≥ exit_target_ts
- compute:
  - exit_mid = (best_bid + best_ask) / 2
  - exit_ts = actual order book timestamp used

Important:
- exit uses order book mid
- exit does not use trade price

--------------------------------------------------
9. PnL Calculation

If filled and exited, compute realized PnL in basis points.

Bid:
- pnl_bps = 0.5 + ((exit_mid - entry_price) / entry_mid) * 10000

Ask:
- pnl_bps = 0.5 + ((entry_price - exit_mid) / entry_mid) * 10000

Where:
- entry_mid = mid_t
- maker rebate = +0.5 bps

--------------------------------------------------
10. Required Output Philosophy

The output should contain all raw information needed for later performance evaluation.

This script should NOT summarize results.  
It should only write detailed raw backtest records.

The output should support later computation of:
- trade count
- fill rate
- total pnl
- Sharpe ratio
- Calmar ratio
- drawdown
- pnl by day / hour / side / offset
- execution filter quality

--------------------------------------------------
11. Output Granularity

The script writes one row per (decision timestamp, side) pair.

At each decision timestamp t, up to two rows may be written:
- one for bid side
- one for ask side

If a side is in cooldown at timestamp t, no row is written for that side.

Each row records one of the following outcomes:
- skipped by execution model (execute_decision = 0)
- executed but not filled (execute_decision = 1, fill_flag = 0)
- filled and exited (execute_decision = 1, fill_flag = 1)

This is better than writing only filled trades, because later performance analysis may need to study:
- fill rate
- execution rate
- selection bias
- no-trade intervals
- cooldown frequency

--------------------------------------------------
12. Required Output Fields

Each output row should contain enough information for later evaluation.

Fields:

Core identifiers
- ts
- date
- hour_of_day
- side

Market state / features
- OBI5
- OBI25
- OBI400
- NTR_10s
- mid_std_2s
- spread_bps
- trade_flow_10s
- trade_count_10s
- cumulativeVolume_5bps
- best_bid
- best_ask
- mid_t

Model outputs
- offset_model_raw_prediction
- predicted_offset_bps
- execute_model_raw_prediction   (0/1 from model.predict)
- execute_model_proba            (P(class=1) from model.predict_proba)
- execute_decision               (same as raw_prediction; 1 = execute)

Quote details
- quote_price
- quote_active_start_ts
- quote_active_end_ts

Execution / fill outcomes
- fill_flag
- fill_ts
- fill_side
- fill_size_used
- fill_price

Exit outcomes
- exit_flag
- exit_target_ts
- exit_ts
- exit_mid

PnL / trade result
- pnl = pnl_bps/10000
- pnl_bps
- valid_trade_flag

--------------------------------------------------
13. Meaning of Output Cases

Case 1: skipped by execution model
- execute_decision = 0
- no quote placed
- fill_flag = 0
- pnl_bps = NA

Case 2: executed but no fill
- execute_decision = 1
- quote placed
- fill_flag = 0
- pnl_bps = NA
Case 3: executed and filled
- execute_decision = 1
- fill_flag = 1
- exit_flag = 1 if exit resolved
- pnl_bps computed

This structure allows later evaluation scripts to decide whether no-fill timestamps count as zero return or should be analyzed separately.
Keep all rows, do not delete rows
--------------------------------------------------
14. Modularity Requirements

The script should be modularized into components such as:
- model loader
- order book replay engine
- trade loader / trade buffer
- feature engine
- inference engine
- execution simulator
- output writer

This allows future changes such as:
- swapping models
- changing feature definitions
- changing execution rules
- adding alternative fill logic

--------------------------------------------------
15. Key Consistency Requirement

The deployment/backtest logic must stay consistent with the training data generation pipeline:

- same feature definitions
- same feature timing rules
- same quote construction
- same fill logic
- same exit logic
- same pnl formula

The only difference is:
- in training, labels were generated from oracle best offset
- in deployment, decisions come from trained models

--------------------------------------------------
16. End Goal

Produce a raw backtesting output dataset that records model decisions and realized trade outcomes at each decision timestamp, with enough detail to compute all performance metrics later in a separate evaluation script.

17. Notes:
- the script will run both side (bid and ask) at the same time, which is both sides in a single pass
- for the overlapping position:
When a quote is placed on a side, skip new decisions for that side until the current trade fully resolves (exit or no-fill). Specifically:
No fill → side is free again at t + quote_active_ms (1s)
Filled → side is free again at exit_ts
"""

from __future__ import annotations

import argparse
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse all shared logic from step1
from step1 import (
    UTC,
    ORDER_BOOK_DIR,
    TRADE_DIR,
    PipelineConfig,
    TradeData,
    SampleState,
    log_progress,
    load_pyarrow,
    date_to_datetime,
    datetime_to_ms,
    ms_to_datetime,
    day_bounds_ms,
    previous_date,
    next_date,
    order_book_path,
    trade_path,
    available_dates,
    ensure_dir,
    safe_div,
    compute_obi,
    compute_mid_std,
    apply_levels,
    rebuild_book,
    current_mid_from_book,
    book_metrics,
    iter_order_book_records,
    load_trade_data_for_day,
    build_grid_timestamps,
    trade_window_bounds,
    compute_ntr_bps,
    find_first_fill,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "OBI5", "OBI25", "OBI400", "NTR_10s", "mid_std_2s",
    "spread_bps", "hour_of_day", "trade_flow_10s",
    "trade_count_10s", "cumulativeVolume_5bps",
]

# Class index → offset in bps (must match step2_1.py training)
OFFSET_CLASS_MAP: Tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.25,
)

DEFAULT_OUTPUT_ROOT = Path("/Users/luke/Desktop/project_intern/result/step3_classifier_full")
DEFAULT_MODEL_DIR = Path("/Users/luke/Desktop/project_intern/code")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_xgb_classifier(path: Path):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def _load_xgb_regressor(path: Path):
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(str(path))
    return model


@dataclass
class ModelSet:
    """Holds offset + execution models for one side."""
    offset_model: object
    execution_model: object
    side: str
    offset_class_map: Tuple[float, ...] = OFFSET_CLASS_MAP


def load_models(model_dir: Path) -> Dict[str, ModelSet]:
    """Load bid and ask model pairs."""
    models: Dict[str, ModelSet] = {}
    for side in ("bid", "ask"):
        # change this for changing model
        offset_path = model_dir / f"xgboost_multiClassification_{side}.json"
        # offset_path = model_dir / f"xgboost_regressor_{side}.json"
        exec_path = model_dir / f"xgboost_multiClassification_netPnL_{side}.json"
        if not offset_path.exists():
            raise FileNotFoundError(f"Offset model not found: {offset_path}")
        if not exec_path.exists():
            raise FileNotFoundError(f"Execution model not found: {exec_path}")
        models[side] = ModelSet(
            # change this for changing model
            offset_model=_load_xgb_classifier(offset_path),
            # offset_model=_load_xgb_regressor(offset_path),
            execution_model=_load_xgb_classifier(exec_path),
            side=side,
        )
        log_progress(f"loaded models for {side}: {offset_path.name}, {exec_path.name}")
    return models


# ---------------------------------------------------------------------------
# Feature extraction from SampleState
# ---------------------------------------------------------------------------


def sample_to_feature_array(sample: SampleState) -> np.ndarray:
    """Extract the 10 features in the same order used during training."""
    return np.array([[
        sample.OBI5,
        sample.OBI25,
        sample.OBI400,
        sample.NTR_10s,
        sample.mid_std_2s,
        sample.spread_bps,
        float(sample.hour_of_day),
        sample.trade_flow_10s,
        float(sample.trade_count_10s),
        sample.depth_5bps,
    ]])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# change this for changing model
def predict_offset(model_set: ModelSet, X: np.ndarray) -> Tuple[int, float]:
    """Return (raw_class_prediction, offset_bps)."""
    raw_pred = int(model_set.offset_model.predict(X)[0])
    offset_bps = model_set.offset_class_map[raw_pred]
    return raw_pred, offset_bps

# def predict_offset(model_set: ModelSet, X: np.ndarray) -> Tuple[float, float]:
#     raw_pred = float(model_set.offset_model.predict(X)[0])
#     offset_bps = raw_pred
#     return raw_pred, offset_bps

def predict_execute(model_set: ModelSet, X: np.ndarray) -> Tuple[float, int]:
    """Return (raw_prediction_proba_class1, execute_decision)."""
    raw_pred = int(model_set.execution_model.predict(X)[0])
    # class 1 = positive PnL → execute, class 0 → skip
    proba = float(model_set.execution_model.predict_proba(X)[0, 1])
    return proba, raw_pred


# ---------------------------------------------------------------------------
# Exit helpers
# ---------------------------------------------------------------------------


def _apply_exit_to_row(
    config: PipelineConfig,
    row: Dict,
    side: str,
    exit_ts: int,
    exit_mid: float,
) -> None:
    """Finalize one filled row once the exit OB event is known."""
    row["exit_flag"] = 1
    row["exit_ts"] = exit_ts
    row["exit_mid"] = exit_mid
    row["valid_trade_flag"] = 1

    entry_price = row["fill_price"]
    entry_mid = row["mid_t"]
    if side == "bid":
        pnl_bps = config.maker_rebate_bps + ((exit_mid - entry_price) / entry_mid) * 10_000.0
    else:
        pnl_bps = config.maker_rebate_bps + ((entry_price - exit_mid) / entry_mid) * 10_000.0

    row["pnl_bps"] = pnl_bps
    row["pnl"] = pnl_bps / 10_000.0


def _resolve_pending_exits_at_event(
    config: PipelineConfig,
    output_rows: List[Dict],
    pending_exit_by_side: Dict[str, Optional[Tuple[int, int]]],
    event_ts: int,
    event_mid: float,
    cooldown_until: Dict[str, int],
) -> int:
    """
    Resolve any open filled trades whose exit_target_ts has been reached by the
    current order-book event. This keeps side cooldowns accurate during replay.
    """
    resolved = 0
    for side in ("bid", "ask"):
        pending = pending_exit_by_side.get(side)
        if pending is None:
            continue

        row_idx, exit_target_ts = pending
        if exit_target_ts > event_ts:
            continue

        _apply_exit_to_row(config, output_rows[row_idx], side, event_ts, event_mid)
        cooldown_until[side] = max(cooldown_until.get(side, 0), event_ts)
        pending_exit_by_side[side] = None
        resolved += 1

    return resolved


# ---------------------------------------------------------------------------
# Single-pass backtest for one day
# ---------------------------------------------------------------------------


def backtest_day(
    config: PipelineConfig,
    date_str: str,
    models: Dict[str, ModelSet],
    trades: TradeData,
) -> List[Dict]:
    """
    Replay one day: build sample states via OB replay (same as step1),
    then for each grid timestamp run model inference, fill simulation, exit resolution.
    """
    path = order_book_path(config, date_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing order book file: {path}")

    grid_timestamps = build_grid_timestamps(config, date_str)
    log_progress(
        f"{date_str}: backtest starting | grid_points={len(grid_timestamps)}"
    )

    # --- State for OB replay (mirrors step1.build_sample_states) ---
    mid_window: Deque[Tuple[int, float]] = deque()
    bid_book: Dict[float, float] = {}
    ask_book: Dict[float, float] = {}
    current_state_ready = False
    next_grid_idx = 0
    day_start_ms, day_end_ms = day_bounds_ms(date_str)
    record_count = 0
    earliest_trade_ts = trades.timestamps[0] if len(trades) > 0 else None

    # --- Cooldown tracking per side ---
    cooldown_until: Dict[str, int] = {"bid": 0, "ask": 0}

    # --- Open filled trades per side: (output_row_index, exit_target_ts) ---
    pending_exit_by_side: Dict[str, Optional[Tuple[int, int]]] = {"bid": None, "ask": None}

    # --- Output rows ---
    output_rows: List[Dict] = []

    def process_grid_ts(sample_ts: int) -> None:
        # Prune mid window
        while mid_window and mid_window[0][0] < sample_ts - config.mid_std_window_ms:
            mid_window.popleft()

        metrics = book_metrics(bid_book, ask_book, config.depth_band_bps)
        if metrics is None:
            return

        # Feature readiness (same as step1)
        trade_window_ready = (earliest_trade_ts is not None
                              and earliest_trade_ts <= sample_ts - config.lookback_ms)
        mid_window_ready = len(mid_window) >= 2
        if not (trade_window_ready and mid_window_ready):
            return

        window_start_ts = sample_ts - config.lookback_ms
        start_idx, end_idx = trade_window_bounds(trades, window_start_ts, sample_ts)
        dt = ms_to_datetime(sample_ts)

        sample = SampleState(
            ts=sample_ts,
            date=dt.strftime("%Y-%m-%d"),
            hour_of_day=dt.hour,
            best_bid=metrics["best_bid"],
            best_ask=metrics["best_ask"],
            mid=metrics["mid"],
            OBI5=metrics["OBI5"],
            OBI25=metrics["OBI25"],
            OBI400=metrics["OBI400"],
            NTR_10s=compute_ntr_bps(trades, window_start_ts, start_idx, end_idx, metrics["mid"]),
            mid_std_2s=compute_mid_std(mid_window),
            spread_bps=metrics["spread_bps"],
            trade_flow_10s=trades.trade_flow(start_idx, end_idx),
            trade_count_10s=end_idx - start_idx,
            depth_5bps=metrics["depth_5bps"],
            feature_window_ready=True,
        )

        # Check for NaN in features — skip if present
        X = sample_to_feature_array(sample)
        if np.any(np.isnan(X)):
            return

        # --- Run both sides ---
        for side in ("bid", "ask"):
            if pending_exit_by_side[side] is not None:
                continue  # side has an open filled trade; wait until exit resolves
            if sample_ts < cooldown_until[side]:
                continue  # side in cooldown, skip (no row written)

            model_set = models[side]

            # Offset model
            offset_raw, offset_bps = predict_offset(model_set, X)

            # Execution model
            exec_proba, exec_decision = predict_execute(model_set, X)

            # Base row (always written)
            row = {
                "ts": sample_ts,
                "date": sample.date,
                "hour_of_day": sample.hour_of_day,
                "side": side,
                "OBI5": sample.OBI5,
                "OBI25": sample.OBI25,
                "OBI400": sample.OBI400,
                "NTR_10s": sample.NTR_10s,
                "mid_std_2s": sample.mid_std_2s,
                "spread_bps": sample.spread_bps,
                "trade_flow_10s": sample.trade_flow_10s,
                "trade_count_10s": sample.trade_count_10s,
                "cumulativeVolume_5bps": sample.depth_5bps,
                "best_bid": sample.best_bid,
                "best_ask": sample.best_ask,
                "mid_t": sample.mid,
                "offset_model_raw_prediction": offset_raw,
                "predicted_offset_bps": offset_bps,
                "execute_model_raw_prediction": exec_decision,
                "execute_model_proba": exec_proba,
                "execute_decision": exec_decision,
                "quote_price": None,
                "quote_active_start_ts": None,
                "quote_active_end_ts": None,
                "fill_flag": 0,
                "fill_ts": None,
                "fill_side": side,
                "fill_size_used": None,
                "fill_price": None,
                "exit_flag": 0,
                "exit_target_ts": None,
                "exit_ts": None,
                "exit_mid": None,
                "pnl": None,
                "pnl_bps": None,
                "valid_trade_flag": 0,
            }

            if exec_decision == 0:
                output_rows.append(row)
                continue

            # --- Quote placement ---
            if side == "bid":
                quote_price = sample.mid * (1.0 - offset_bps / 10_000.0)
            else:
                quote_price = sample.mid * (1.0 + offset_bps / 10_000.0)

            row["quote_price"] = quote_price
            row["quote_active_start_ts"] = sample_ts
            row["quote_active_end_ts"] = sample_ts + config.quote_active_ms

            # --- Fill simulation (same logic as step1) ---
            fill_start_idx = bisect_right(trades.timestamps, sample_ts)
            fill_end_idx = bisect_right(trades.timestamps, sample_ts + config.quote_active_ms)

            fill_ts = find_first_fill(
                trades, fill_start_idx, fill_end_idx,
                side, quote_price, config.min_fill_size,
            )

            if fill_ts is None:
                # No fill → cooldown ends at quote_active_end
                cooldown_until[side] = sample_ts + config.quote_active_ms
                output_rows.append(row)
                continue

            # --- Filled ---
            row["fill_flag"] = 1
            row["fill_ts"] = fill_ts
            row["fill_price"] = quote_price  # fill price = quote price
            row["fill_size_used"] = config.min_fill_size
            row["exit_target_ts"] = fill_ts + config.exit_delay_ms

            row_idx = len(output_rows)
            output_rows.append(row)
            pending_exit_by_side[side] = (row_idx, fill_ts + config.exit_delay_ms)

    # --- Main OB replay loop (same structure as step1) ---
    for record in iter_order_book_records(path, config.instrument):
        record_count += 1
        event_ts = int(record["ts"])

        # Process grid timestamps before this OB event
        while (current_state_ready
               and next_grid_idx < len(grid_timestamps)
               and grid_timestamps[next_grid_idx] < event_ts
               and grid_timestamps[next_grid_idx] < day_end_ms):
            process_grid_ts(grid_timestamps[next_grid_idx])
            next_grid_idx += 1

        rebuild_book(
            action=str(record.get("action", "")),
            bids=record.get("bids", []),
            asks=record.get("asks", []),
            bid_book=bid_book,
            ask_book=ask_book,
        )
        current_state_ready = True

        # Track event-level mid for mid_std_2s
        event_mid = current_mid_from_book(bid_book, ask_book)
        if event_mid is not None:
            mid_window.append((event_ts, event_mid))
            _resolve_pending_exits_at_event(
                config,
                output_rows,
                pending_exit_by_side,
                event_ts,
                event_mid,
                cooldown_until,
            )

        # Process grid timestamps at this OB event
        while (current_state_ready
               and next_grid_idx < len(grid_timestamps)
               and grid_timestamps[next_grid_idx] == event_ts
               and grid_timestamps[next_grid_idx] < day_end_ms):
            process_grid_ts(grid_timestamps[next_grid_idx])
            next_grid_idx += 1

        if record_count % 1_000_000 == 0:
            log_progress(f"{date_str}: processed {record_count} OB records | rows={len(output_rows)}")

    # Process remaining grid timestamps
    while (current_state_ready
           and next_grid_idx < len(grid_timestamps)
           and grid_timestamps[next_grid_idx] < day_end_ms):
        process_grid_ts(grid_timestamps[next_grid_idx])
        next_grid_idx += 1

    log_progress(
        f"{date_str}: OB replay done | records={record_count} | rows={len(output_rows)} "
        f"| pending_exits={sum(1 for pending in pending_exit_by_side.values() if pending is not None)}"
    )

    pending_exits = [
        (row_idx, exit_target_ts, side)
        for side, pending in pending_exit_by_side.items()
        if pending is not None
        for row_idx, exit_target_ts in (pending,)
    ]
    resolve_exits_from_events(config, date_str, output_rows, pending_exits)

    log_progress(f"{date_str}: backtest done | output_rows={len(output_rows)}")
    return output_rows


def resolve_exits_from_events(
    config: PipelineConfig,
    date_str: str,
    output_rows: List[Dict],
    pending_exits: List[Tuple[int, int, str]],
) -> None:
    """Resolve exits that remain open after day-end using next day's OB events."""
    if not pending_exits:
        return

    log_progress(f"{date_str}: resolving {len(pending_exits)} overnight exits from next day OB")
    next_day = next_date(date_str)
    next_path = order_book_path(config, next_day)
    resolved = 0

    if next_path.exists():
        bid_book: Dict[float, float] = {}
        ask_book: Dict[float, float] = {}
        pending_idx = 0
        pending_exits = sorted(pending_exits, key=lambda x: x[1])

        for record in iter_order_book_records(next_path, config.instrument):
            rebuild_book(
                action=str(record.get("action", "")),
                bids=record.get("bids", []),
                asks=record.get("asks", []),
                bid_book=bid_book,
                ask_book=ask_book,
            )
            mid = current_mid_from_book(bid_book, ask_book)
            if mid is None:
                continue

            event_ts = int(record["ts"])
            while pending_idx < len(pending_exits) and pending_exits[pending_idx][1] <= event_ts:
                row_idx, _, side = pending_exits[pending_idx]
                _apply_exit_to_row(config, output_rows[row_idx], side, event_ts, mid)
                pending_idx += 1
                resolved += 1

            if pending_idx >= len(pending_exits):
                break

    unresolved = sum(1 for row_idx, _, _ in pending_exits if output_rows[row_idx]["exit_flag"] == 0)
    log_progress(f"{date_str}: exit resolution done | resolved={resolved} | unresolved={unresolved}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def backtest_schema():
    pa, _ = load_pyarrow()
    return pa.schema([
        ("ts", pa.int64()),
        ("date", pa.string()),
        ("hour_of_day", pa.int32()),
        ("side", pa.string()),
        ("OBI5", pa.float64()),
        ("OBI25", pa.float64()),
        ("OBI400", pa.float64()),
        ("NTR_10s", pa.float64()),
        ("mid_std_2s", pa.float64()),
        ("spread_bps", pa.float64()),
        ("trade_flow_10s", pa.float64()),
        ("trade_count_10s", pa.int64()),
        ("cumulativeVolume_5bps", pa.float64()),
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("mid_t", pa.float64()),
        ("offset_model_raw_prediction", pa.float64()),
        ("predicted_offset_bps", pa.float64()),
        ("execute_model_raw_prediction", pa.int32()),
        ("execute_model_proba", pa.float64()),
        ("execute_decision", pa.int32()),
        ("quote_price", pa.float64()),
        ("quote_active_start_ts", pa.int64()),
        ("quote_active_end_ts", pa.int64()),
        ("fill_flag", pa.int32()),
        ("fill_ts", pa.int64()),
        ("fill_side", pa.string()),
        ("fill_size_used", pa.float64()),
        ("fill_price", pa.float64()),
        ("exit_flag", pa.int32()),
        ("exit_target_ts", pa.int64()),
        ("exit_ts", pa.int64()),
        ("exit_mid", pa.float64()),
        ("pnl", pa.float64()),
        ("pnl_bps", pa.float64()),
        ("valid_trade_flag", pa.int32()),
    ])


def write_backtest_parquet(path: Path, rows: List[Dict]) -> None:
    pa, pq = load_pyarrow()
    ensure_dir(path.parent)
    log_progress(f"writing {path} | rows={len(rows)}")
    table = pa.Table.from_pylist(rows, schema=backtest_schema())
    pq.write_table(table, path)


def write_combined_backtest(output_root: Path, daily_paths: Sequence[Path]) -> Path:
    _, pq = load_pyarrow()
    combined_path = output_root / "backtest_combined.parquet"
    log_progress(f"combining {len(daily_paths)} daily files into {combined_path}")
    writer = pq.ParquetWriter(combined_path, backtest_schema())
    try:
        for p in daily_paths:
            writer.write_table(pq.read_table(p))
    finally:
        writer.close()
    log_progress(f"combined backtest written: {combined_path}")
    return combined_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trained models on BTC-USDT-SWAP.")
    parser.add_argument("--dates", nargs="*", help="Dates to backtest, e.g. 2026-01-01 2026-01-02.")
    parser.add_argument("--order-book-dir", type=Path, default=ORDER_BOOK_DIR)
    parser.add_argument("--trade-dir", type=Path, default=TRADE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--skip-combine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        order_book_dir=args.order_book_dir,
        trade_dir=args.trade_dir,
        output_root=args.output_root,
        sampling_ms=1_000,
        samples_per_day=None,  # backtest uses full grid (all 86400 timestamps)
        lookback_ms=10_000,
        quote_active_ms=1_000,
        exit_delay_ms=1_000,
        min_fill_size=1.0,
    )

    models = load_models(args.model_dir)

    dates = args.dates if args.dates else available_dates(config)
    if not dates:
        raise FileNotFoundError(f"No OB files found in {config.order_book_dir}")

    daily_dir = args.output_root / "daily"
    ensure_dir(daily_dir)

    log_progress(f"starting backtest | dates={len(dates)} | output={args.output_root}")
    daily_paths: List[Path] = []
    total_rows = 0

    for date_str in dates:
        trades = load_trade_data_for_day(config, date_str)
        rows = backtest_day(config, date_str, models, trades)
        daily_path = daily_dir / f"backtest_{date_str}.parquet"
        write_backtest_parquet(daily_path, rows)
        daily_paths.append(daily_path)
        total_rows += len(rows)

        filled = sum(1 for r in rows if r["fill_flag"] == 1)
        executed = sum(1 for r in rows if r["execute_decision"] == 1)
        print(f"[step3] {date_str} | rows={len(rows)} | executed={executed} | filled={filled}")

    if not args.skip_combine and daily_paths:
        combined = write_combined_backtest(args.output_root, daily_paths)
        print(f"[step3] combined={combined} | total_rows={total_rows}")

    log_progress(f"backtest complete | total_rows={total_rows} | days={len(daily_paths)}")


if __name__ == "__main__":
    main()
