"""
Project: Side-Separated Quote-Offset Labeling Dataset for BTC-USDT-SWAP

Overview
This pipeline constructs a supervised learning dataset for quote-offset prediction on BTC-USDT-SWAP using OKX order book data and trade data.

The output is a labeled dataset where each row represents one side-specific quoting opportunity:
- bid side, or
- ask side

For each sampled timestamp t, the pipeline:
1. computes market-state features X using only information available at or before t
2. evaluates a grid of candidate quote offsets for the bid side and ask side separately
3. simulates fills within a short quote-active window
4. exits 1 second after fill using the first available order-book midprice
5. computes rebate-adjusted PnL
6. selects the best offset for that side
7. writes one labeled row per valid side

This step only constructs the dataset.
It does not perform train/validation/test split.
It does not train any model.

--------------------------------------------------
1. Objective

Build a side-separated labeled dataset for BTC-USDT-SWAP where each row contains:
- market-state features at decision time t
- side = bid or ask
- the optimal quote offset for that side
- the corresponding realized PnL
- fill and exit timestamps

The dataset is intended for later supervised learning.

--------------------------------------------------
2. Instrument and Data Sources

Instrument
- BTC-USDT-SWAP

Order book directory
- /Users/luke/Desktop/project_intern/data/data_ob

Trade directory
- /Users/luke/Desktop/project_intern/data/data_trade

--------------------------------------------------
3. Raw Data Formats

3.1 Order Book Files
File naming pattern:
- BTC-USDT-SWAP-L2orderbook-400lv-YYYY-MM-DD.data

Each line in a .data file is one JSON object.

Expected keys:
- instId
- action
- ts
- asks
- bids

Example:
{
  "instId": "BTC-USDT-SWAP",
  "action": "update",
  "ts": "1767225600014",
  "asks": [["87619.2", "21.41", "1"], ...],
  "bids": [["87588.4", "0.62", "2"], ...]
}

Interpretation:
- instId: instrument identifier
- action: "snapshot" or "update"
- ts: Unix timestamp in milliseconds
- asks: ask-side book levels
- bids: bid-side book levels

Each level is:
- [price, size, order_count]

In this project:
- price and size are used
- order_count is ignored

Order book reconstruction:
- snapshot replaces the current in-memory book
- update applies incremental changes to the current in-memory book
- levels with size <= 0 are removed

3.2 Trade Files
File naming pattern:
- BTC-USDT-SWAP-trades-YYYY-MM-DD.csv

Expected columns:
- instrument_name
- trade_id
- side
- price
- size
- created_time

Interpretation:
- instrument_name should be BTC-USDT-SWAP
- created_time is Unix timestamp in milliseconds
- side is used as aggressor side:
  - buy -> aggressor_sign = +1
  - sell -> aggressor_sign = -1
- price is trade price
- size is trade size

--------------------------------------------------
4. Core Modeling Assumptions

4.1 Order Size
- Fixed order size = 1 unit
- The size unit is the same as the size unit in the raw order book and trade data

4.2 Fill Logic
- No partial fills
- A quote is considered filled only if cumulative qualifying trade size reaches at least 1 unit

4.3 Candidate Quote Offsets
The pipeline evaluates the following candidate offsets in basis points:
- 0.0
- 0.25
- 0.5
- 0.75
- 1.0
- 1.25
- 1.5
- 1.75
- 2.0
- 3.0

4.4 Rebate
- Maker rebate = +0.5 bps
- Rebate is included directly in the PnL
- Optimal quote selection is based on PnL including rebate

--------------------------------------------------
5. Sampling Rule

The pipeline does not build one row for every raw event timestamp.

Instead, for each day:
1. build a fixed time grid using sampling_ms
2. optionally subsample timestamps from that grid
3. evaluate only those sampled timestamps

Default configuration:
- sampling_ms = 1000 (1 second)
- samples_per_day = 6000
- random_subsample = False

Sampling behavior:
- if random_subsample = False:
  choose an evenly spaced subset of the daily grid
- if random_subsample = True:
  randomly sample timestamps from the daily grid using a deterministic seed

This avoids sampling directly from irregular raw event timestamps.

--------------------------------------------------
6. Strict Time Alignment

6.1 Feature Construction
Features at time t use only data at or before t.

This includes:
- order book state at sampled timestamp t
- rolling mid-price history over [t-2s, t] for mid_std_2s
- trades in the lookback window [t - 10s, t]

6.2 Label Construction
Labels use only future information after t.

This includes:
- fill search in (t, t+1s]
- exit at first available order-book mid at or after fill_ts + 1s

No future information is used in X.

--------------------------------------------------
7. Feature Set (X)

For each sampled timestamp t, the pipeline computes:

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

7.1 Feature Definitions

OBI5
- top 5 bid depth and top 5 ask depth are summed
- OBI5 = (bid_size_top5 - ask_size_top5) / (bid_size_top5 + ask_size_top5)

OBI25
- top 25 bid depth and top 25 ask depth are summed
- OBI25 = (bid_size_top25 - ask_size_top25) / (bid_size_top25 + ask_size_top25)


OBI400
- top 400 bid depth and top 400 ask depth are summed
- OBI400 = (bid_size_top400 - ask_size_top400) / (bid_size_top400 + ask_size_top400)

NTR_10s
-computed from trades in the window [t − 10s, t]
-let:
-window_high = maximum trade price in the window
-window_low = minimum trade price in the window
-prev_close = last trade price strictly before the window, which is the first price before t-10s; if no trades exist before t-10s, return NA

-compute true range:
-TR_10s = max(
-window_high - window_low,
-abs(window_high - prev_close),
-abs(window_low - prev_close))
-normalize to basis points using mid price at time t:
-NTR_10s = (TR_10s / mid_t) * 10000
-if no trades exist in the window, return NA

mid_std_2s
-computed from order-book mid prices in the window [t − 2s, t]
-mid is defined as:

-mid = (best_bid + best_ask) / 2
-collect all mid values from order-book events within the window
-compute standard deviation:

-mid_std_2s = sqrt( (1/N) * sum( (mid_i - mean(mid))^2 ) )
-if fewer than 2 mid values exist, return NA

spread_bps
- mid = (best_bid + best_ask) / 2
- spread_bps = ((best_ask - best_bid) / mid) * 10000

hour_of_day
- UTC hour extracted from sampled timestamp t

trade_flow_10s
-net signed trade size over [t − 10s, t]
-buy trades contribute positive size
-sell trades contribute negative size
-trade_flow_10s = sum(size * sign)

trade_count_10s
- number of trades in [t - 10s, t]

cumulativeVolume_5bps (formerlly in this code called depth_5bps)
-total visible order book size within ±5 bps of mid_t
-define price band:
-bid side: price ≥ mid_t * (1 − 5bps)
-ask side: price ≤ mid_t * (1 + 5bps)
-compute:
-bid_depth_5bps = sum of bid sizes within band
-ask_depth_5bps = sum of ask sizes within band
-cumulativeVolume_5bps = bid_depth_5bps + ask_depth_5bps


--------------------------------------------------
8. Order Book Filtering / Bad Timestamp Handling

A sample is ignored if order-book state is invalid.

Current code filters out samples where:
- bid book is empty
- ask book is empty
- best_ask <= best_bid
- spread_bps <= 0
- spread_bps > 50

A sampled timestamp can still be dropped later if:
- feature window is incomplete:
  - trade lookback requires that the earliest available trade timestamp is at or before t - 10s
  - mid_std_2s requires at least 2 order-book event mids within [t - 2s, t]
- no valid fill occurs for a side
- no valid exit price is found for a candidate

--------------------------------------------------
9. Quote Construction

For each sampled timestamp t and each candidate offset:

Let:
- mid_t = current mid price at time t

Bid quote:
- bid_quote = mid_t * (1 - offset_bps / 10000)

Ask quote:
- ask_quote = mid_t * (1 + offset_bps / 10000)

Interpretation:
- smaller offset = more aggressive quote
- larger offset = more conservative quote

Quotes are defined relative to mid_t.

--------------------------------------------------
10. Quote Active Window

For each sampled timestamp t:
- quote-active window = (t, t + 1 second]

This is controlled by:
- quote_active_ms = 1000

Only fills within this window are considered valid.

--------------------------------------------------
11. Fill Logic

11.1 Bid Side Fill
A bid quote is considered filled if, within (t, t+1s]:
- the trade is seller-initiated
- trade price <= bid_quote
- cumulative qualifying trade size reaches at least 1

The fill timestamp is the first timestamp where cumulative qualifying size >= 1.

11.2 Ask Side Fill
An ask quote is considered filled if, within (t, t+1s]:
- the trade is buyer-initiated
- trade price >= ask_quote
- cumulative qualifying trade size reaches at least 1

The fill timestamp is the first timestamp where cumulative qualifying size >= 1.

Notes:
- no partial fills are recorded
- only the first full-fill timestamp is kept
- fill price is assumed to equal the quoted price

--------------------------------------------------
12. Exit Logic

If a quote fills at time x:
- exit_target_ts = x + 1 second

The pipeline then looks through order-book updates and uses:
- the first available order-book snapshot at or after exit_target_ts

From that snapshot, it computes:
- exit_mid = (best_bid + best_ask) / 2
- exit_ts = actual order-book event timestamp used for exit

This exit logic is the same for both bid-side and ask-side candidates.

Important:
- exit uses order-book mid
- exit does NOT use first available trade price
- exit does NOT simulate a second maker or taker execution

--------------------------------------------------
13. PnL Definition

PnL is computed in basis points and includes the maker rebate.

13.1 Bid-Side PnL
For a filled bid quote:
- entry = buy at bid_quote
- exit = sell at exit_mid

PnL_bid_bps =
0.5 + ((exit_mid - entry_price) / entry_mid) * 10000

where:
- entry_price = bid_quote
- entry_mid = mid_t
- rebate = +0.5 bps

13.2 Ask-Side PnL
For a filled ask quote:
- entry = sell at ask_quote
- exit = buy back at exit_mid

PnL_ask_bps =
0.5 + ((entry_price - exit_mid) / entry_mid) * 10000

where:
- entry_price = ask_quote
- entry_mid = mid_t
- rebate = +0.5 bps

Interpretation:
- bid-side PnL measures short-horizon long markout plus rebate
- ask-side PnL measures short-horizon short markout plus rebate

--------------------------------------------------
14. Optimal Offset Selection

For each sampled timestamp t and each side independently:

1. evaluate all candidate offsets
2. keep only candidates with:
   - valid fill
   - valid exit price
3. compute PnL including rebate
4. choose the candidate with the highest PnL

Rules:
- if all valid candidate PnLs are negative, choose the smallest loss
- if multiple offsets have the same PnL, choose the smaller offset
- if no candidate is valid for that side, produce no row for that side

This yields side-specific optimal offsets:
- bid side optimal offset
- ask side optimal offset

--------------------------------------------------
15. Side Separation and Row Generation

Each sampled timestamp can produce:
- 2 rows: one bid row and one ask row
- 1 row: only one side valid
- 0 rows: neither side valid

Each final dataset row corresponds to exactly one side-specific opportunity.

The code stores:
- side = "bid" or "ask"
- optimal_offset_bps
- optimal_pnl_bps
- fill_ts
- exit_ts
- entry_price
- exit_mid

--------------------------------------------------
16. Final Output Schema

Each row contains:

-ts
-date
-side
-OBI5
-OBI25
-OBI400
-NTR_10s
-mid_std_2s
-spread_bps
-hour_of_day
-trade_flow_10s
-trade_count_10s
-cumulativeVolume_5bps
-optimal_offset_bps
-optimal_pnl_bps
-fill_ts
-exit_ts
-entry_price
-exit_mid
-valid_label_flag

valid_label_flag is always 1 for written rows because invalid rows are skipped.

--------------------------------------------------
17. Processing Workflow

For each day:
1. load relevant trade data
   - previous day
   - current day
   - next day
   within the required time bounds
2. reconstruct the order book from the current day .data file
3. build sampled market states on the fixed grid
4. build fill candidates for each side and offset
5. resolve exit mid prices from order book events
6. choose best candidate per (sample timestamp, side)
7. write daily labeled output

After all daily files are written:
- optionally combine them into one monthly dataset

--------------------------------------------------
18. Output Files

Per day:
- labeled_side_rows_YYYY-MM-DD.parquet

Combined output:
- labeled_dataset_btc_usdt_swap_side_separated.parquet

Output location:
- /Users/luke/Desktop/project_intern/result/step1_side_separated

--------------------------------------------------
19. Memory Strategy

The pipeline is designed to be memory-conscious.

Current design choices:
- process one day at a time
- do not load the whole month into memory
- use arrays for trade storage
- reconstruct order book incrementally from raw updates
- write daily results to compressed CSV
- combine later if needed

--------------------------------------------------
20. End Objective

Produce a day-by-day labeled dataset where:
- each row is one side-specific quote opportunity
- features use only past/current information
- labels use future outcomes only
- fill logic respects 1-unit size constraints
- exit occurs 1 second after fill using first available order-book mid
- optimal offset selection includes rebate
- invalid timestamps and invalid candidates are skipped
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


UTC = timezone.utc

ORDER_BOOK_DIR = Path("/Users/luke/Desktop/project_intern/data/data_ob")
TRADE_DIR = Path("/Users/luke/Desktop/project_intern/data/data_trade")
OUTPUT_ROOT = Path("/Users/luke/Desktop/project_intern/result/step1_5")

DEFAULT_OFFSETS_BPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.25)


@dataclass(frozen=True)
class PipelineConfig:
    instrument: str = "BTC-USDT-SWAP"
    order_book_dir: Path = ORDER_BOOK_DIR
    trade_dir: Path = TRADE_DIR
    output_root: Path = OUTPUT_ROOT
    sampling_ms: int = 1_000
    samples_per_day: Optional[int] = 6000
    random_subsample: bool = False
    random_seed: int = 7
    lookback_ms: int = 10_000
    mid_std_window_ms: int = 2_000
    quote_active_ms: int = 1_000
    exit_delay_ms: int = 1_000
    min_fill_size: float = 1.0
    maker_rebate_bps: float = 0.5
    depth_band_bps: float = 5.0
    candidate_offsets_bps: Tuple[float, ...] = DEFAULT_OFFSETS_BPS


@dataclass
class TradeData:
    timestamps: array = field(default_factory=lambda: array("q"))
    prices: array = field(default_factory=lambda: array("d"))
    sizes: array = field(default_factory=lambda: array("d"))
    aggressor_signs: array = field(default_factory=lambda: array("b"))
    signed_size_prefix: List[float] = field(default_factory=lambda: [0.0])

    def append(self, ts: int, price: float, size: float, aggressor_sign: int) -> None:
        self.timestamps.append(ts)
        self.prices.append(price)
        self.sizes.append(size)
        self.aggressor_signs.append(aggressor_sign)
        self.signed_size_prefix.append(self.signed_size_prefix[-1] + size * aggressor_sign)

    def __len__(self) -> int:
        return len(self.timestamps)

    def trade_flow(self, start_idx: int, end_idx: int) -> float:
        return self.signed_size_prefix[end_idx] - self.signed_size_prefix[start_idx]


@dataclass
class SampleState:
    ts: int
    date: str
    hour_of_day: int
    best_bid: float
    best_ask: float
    mid: float
    OBI5: float
    OBI25: float
    OBI400: float
    NTR_10s: float
    mid_std_2s: float
    spread_bps: float
    trade_flow_10s: float
    trade_count_10s: int
    depth_5bps: float
    feature_window_ready: bool


@dataclass
class FillCandidate:
    sample_ts: int
    side: str
    offset_bps: float
    fill_ts: int
    entry_price: float
    entry_mid: float
    exit_target_ts: int
    exit_ts: Optional[int] = None
    exit_mid: Optional[float] = None


def log_progress(message: str) -> None:
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[step1 {timestamp}] {message}", flush=True)


def load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet output requires pyarrow. Run this script with an environment that has pyarrow installed, "
            "for example ~/venv/0804/bin/python."
        ) from exc
    return pa, pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a side-separated quote-offset dataset for BTC-USDT-SWAP.")
    parser.add_argument("--dates", nargs="*", help="Optional UTC dates to process, e.g. 2026-01-01 2026-01-02.")
    parser.add_argument("--order-book-dir", type=Path, default=ORDER_BOOK_DIR, help="Directory containing .data order book files.")
    parser.add_argument("--trade-dir", type=Path, default=TRADE_DIR, help="Directory containing raw trade CSV files.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Directory for daily and combined outputs.")
    parser.add_argument("--sampling-ms", type=int, default=1_000, help="Base fixed-grid sampling interval in milliseconds.")
    parser.add_argument("--samples-per-day", type=int, default=60000, help="Number of sampled timestamps to keep per day. Use 0 to keep the full grid.")
    parser.add_argument("--random-subsample", action="store_true", help="Randomly subsample the fixed grid instead of taking an evenly spaced subset.")
    parser.add_argument("--random-seed", type=int, default=7, help="Base seed for deterministic random subsampling.")
    parser.add_argument(
        "--lookback-ms",
        type=int,
        default=10_000,
        help="Past window for NTR_10s, trade_flow_10s, and trade_count_10s features.",
    )
    parser.add_argument("--quote-active-ms", type=int, default=1_000, help="Quote active window length after t.")
    parser.add_argument("--exit-delay-ms", type=int, default=1_000, help="Delay from fill timestamp to exit timestamp target.")
    parser.add_argument("--min-fill-size", type=float, default=1.0, help="Minimum cumulative executable size required for a valid fill.")
    parser.add_argument("--skip-combine", action="store_true", help="Skip writing the monthly combined dataset.")
    return parser.parse_args()


def date_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000)


def ms_to_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1_000, tz=UTC)


def day_bounds_ms(date_str: str) -> Tuple[int, int]:
    start = date_to_datetime(date_str)
    end = start + timedelta(days=1)
    return datetime_to_ms(start), datetime_to_ms(end)


def previous_date(date_str: str) -> str:
    return (date_to_datetime(date_str) - timedelta(days=1)).strftime("%Y-%m-%d")


def next_date(date_str: str) -> str:
    return (date_to_datetime(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")


def order_book_path(config: PipelineConfig, date_str: str) -> Path:
    return config.order_book_dir / f"{config.instrument}-L2orderbook-400lv-{date_str}.data"


def trade_path(config: PipelineConfig, date_str: str) -> Path:
    return config.trade_dir / f"{config.instrument}-trades-{date_str}.csv"


def available_dates(config: PipelineConfig) -> List[str]:
    prefix = f"{config.instrument}-L2orderbook-400lv-"
    dates = []
    for path in sorted(config.order_book_dir.glob(f"{config.instrument}-L2orderbook-400lv-*.data")):
        name = path.name
        dates.append(name[len(prefix) : -len(".data")])
    return dates


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_output_dirs(output_root: Path) -> Dict[str, Path]:
    daily_dir = output_root / "daily"
    ensure_dir(daily_dir)
    return {"root": output_root, "daily": daily_dir}


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_obi(bid_size: float, ask_size: float) -> float:
    return safe_div(bid_size - ask_size, bid_size + ask_size)


def compute_mid_std(mid_window: Deque[Tuple[int, float]]) -> float:
    if len(mid_window) < 2:
        return float('nan')

    values = [mid for _, mid in mid_window]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) * (value - mean_value) for value in values) / len(values)
    return math.sqrt(variance)


def apply_levels(levels: Sequence[Sequence[str]], book: Dict[float, float]) -> None:
    for level in levels:
        if len(level) < 2:
            continue
        price = float(level[0])
        size = float(level[1])
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size


def rebuild_book(
    action: str,
    bids: Sequence[Sequence[str]],
    asks: Sequence[Sequence[str]],
    bid_book: Dict[float, float],
    ask_book: Dict[float, float],
) -> None:
    if action == "snapshot":
        bid_book.clear()
        ask_book.clear()
    apply_levels(bids, bid_book)
    apply_levels(asks, ask_book)


def current_mid_from_book(bid_book: Dict[float, float], ask_book: Dict[float, float]) -> Optional[float]:
    if not bid_book or not ask_book:
        return None
    best_bid = max(bid_book)
    best_ask = min(ask_book)
    if best_ask <= best_bid:
        return None
    return (best_bid + best_ask) / 2.0


def book_metrics(
    bid_book: Dict[float, float],
    ask_book: Dict[float, float],
    depth_band_bps: float,
) -> Optional[Dict[str, float]]:
    if not bid_book or not ask_book:
        return None

    bid_levels = sorted(bid_book.items(), key=lambda item: item[0], reverse=True)
    ask_levels = sorted(ask_book.items(), key=lambda item: item[0])
    if not bid_levels or not ask_levels:
        return None

    best_bid, _ = bid_levels[0]
    best_ask, _ = ask_levels[0]
    if best_ask <= best_bid:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid) * 10_000.0
    if spread_bps <= 0:
        return None
    if spread_bps > 50:
        return None

    bid_size_top5 = sum(size for _, size in bid_levels[:5])
    ask_size_top5 = sum(size for _, size in ask_levels[:5])
    bid_size_top25 = sum(size for _, size in bid_levels[:25])
    ask_size_top25 = sum(size for _, size in ask_levels[:25])
    bid_size_top400 = sum(size for _, size in bid_levels[:400])
    ask_size_top400 = sum(size for _, size in ask_levels[:400])

    band = depth_band_bps / 10_000.0
    bid_threshold = mid * (1.0 - band)
    ask_threshold = mid * (1.0 + band)

    bid_depth_5bps = 0.0
    for price, size in bid_levels:
        if price < bid_threshold:
            break
        bid_depth_5bps += size

    ask_depth_5bps = 0.0
    for price, size in ask_levels:
        if price > ask_threshold:
            break
        ask_depth_5bps += size

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "OBI5": compute_obi(bid_size_top5, ask_size_top5),
        "OBI25": compute_obi(bid_size_top25, ask_size_top25),
        "OBI400": compute_obi(bid_size_top400, ask_size_top400),
        "spread_bps": spread_bps,
        "depth_5bps": bid_depth_5bps + ask_depth_5bps,
    }


def iter_order_book_records(path: Path, instrument: str) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("instId") == instrument:
                yield record


def iter_trade_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_trade_data_for_day(config: PipelineConfig, date_str: str) -> TradeData:
    day_start_ms, day_end_ms = day_bounds_ms(date_str)
    lower_bound = day_start_ms - config.lookback_ms
    upper_bound = day_end_ms + config.quote_active_ms + config.exit_delay_ms

    log_progress(
        f"{date_str}: loading trades from {config.trade_dir} for range [{lower_bound}, {upper_bound}]"
    )
    trades = TradeData()
    for candidate_date in (previous_date(date_str), date_str, next_date(date_str)):
        path = trade_path(config, candidate_date)
        if not path.exists():
            continue

        log_progress(f"{date_str}: reading trade file {path.name}")
        kept_rows = 0
        for row in iter_trade_rows(path):
            if row.get("instrument_name") != config.instrument:
                continue

            ts = int(row["created_time"])
            if ts < lower_bound or ts > upper_bound:
                continue

            side = row["side"].strip().lower()
            aggressor_sign = 1 if side == "buy" else -1
            trades.append(
                ts=ts,
                price=float(row["price"]),
                size=float(row["size"]),
                aggressor_sign=aggressor_sign,
            )
            kept_rows += 1

        log_progress(f"{date_str}: kept {kept_rows} trades from {path.name}")

    log_progress(f"{date_str}: finished trade load with {len(trades)} total trades")
    return trades


def build_grid_timestamps(config: PipelineConfig, date_str: str) -> List[int]:
    day_start_ms, day_end_ms = day_bounds_ms(date_str)
    return list(range(day_start_ms, day_end_ms, config.sampling_ms))


def choose_sample_timestamps(config: PipelineConfig, date_str: str, grid_timestamps: Sequence[int]) -> List[int]:
    if not config.samples_per_day or config.samples_per_day <= 0 or config.samples_per_day >= len(grid_timestamps):
        return list(grid_timestamps)

    sample_count = config.samples_per_day
    if config.random_subsample:
        seed = config.random_seed + int(date_str.replace("-", ""))
        rng = random.Random(seed)
        return sorted(rng.sample(list(grid_timestamps), sample_count))

    total = len(grid_timestamps)
    indices = [((2 * i + 1) * total) // (2 * sample_count) for i in range(sample_count)]
    return [grid_timestamps[index] for index in indices]


def trade_window_bounds(trades: TradeData, lower_ts: int, upper_ts: int) -> Tuple[int, int]:
    start_idx = bisect_left(trades.timestamps, lower_ts)
    end_idx = bisect_right(trades.timestamps, upper_ts)
    return start_idx, end_idx


def compute_ntr_bps(
    trades: TradeData,
    window_start_ts: int,
    start_idx: int,
    end_idx: int,
    mid: float,
) -> float:
    if mid <= 0 or start_idx >= end_idx:
        return float('nan')

    window_high = trades.prices[start_idx]
    window_low = trades.prices[start_idx]
    for idx in range(start_idx + 1, end_idx):
        trade_price = trades.prices[idx]
        if trade_price > window_high:
            window_high = trade_price
        if trade_price < window_low:
            window_low = trade_price

    prev_idx = bisect_left(trades.timestamps, window_start_ts) - 1
    if prev_idx >= 0:
        prev_close = trades.prices[prev_idx]
    else:
        return float('nan')

    true_range = max(
        window_high - window_low,
        abs(window_high - prev_close),
        abs(window_low - prev_close),
    )
    return (true_range / mid) * 10_000.0


def build_sample_states(config: PipelineConfig, date_str: str, trades: TradeData) -> List[SampleState]:
    path = order_book_path(config, date_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing order book file: {path}")

    grid_timestamps = build_grid_timestamps(config, date_str)
    selected_timestamps = choose_sample_timestamps(config, date_str, grid_timestamps)
    selected_set = set(selected_timestamps)
    log_progress(
        f"{date_str}: building sample states from {path.name} | grid_points={len(grid_timestamps)} "
        f"| selected_points={len(selected_timestamps)}"
    )

    mid_window: Deque[Tuple[int, float]] = deque()
    sample_rows: List[SampleState] = []

    bid_book: Dict[float, float] = {}
    ask_book: Dict[float, float] = {}
    current_state_ready = False
    next_grid_ts = grid_timestamps[0] if grid_timestamps else None
    day_start_ms, day_end_ms = day_bounds_ms(date_str)
    record_count = 0
    earliest_trade_ts = trades.timestamps[0] if len(trades) > 0 else None

    def maybe_record_sample(sample_ts: int) -> None:
        # Prune event-level mid buffer to keep entries in [sample_ts - mid_std_window_ms, ...]
        while mid_window and mid_window[0][0] < sample_ts - config.mid_std_window_ms:
            mid_window.popleft()

        metrics = book_metrics(bid_book, ask_book, config.depth_band_bps)
        if metrics is None:
            return

        if sample_ts not in selected_set:
            return

        # Feature window readiness: require both full 10s trade window and full 2s mid window
        trade_window_ready = (earliest_trade_ts is not None
                              and earliest_trade_ts <= sample_ts - config.lookback_ms)
        mid_window_ready = len(mid_window) >= 2
        feature_ready = trade_window_ready and mid_window_ready

        window_start_ts = sample_ts - config.lookback_ms
        start_idx, end_idx = trade_window_bounds(trades, window_start_ts, sample_ts)
        dt = ms_to_datetime(sample_ts)

        sample_rows.append(
            SampleState(
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
                feature_window_ready=feature_ready,
            )
        )

    for record in iter_order_book_records(path, config.instrument):
        record_count += 1
        event_ts = int(record["ts"])

        while current_state_ready and next_grid_ts is not None and next_grid_ts < event_ts and next_grid_ts < day_end_ms:
            maybe_record_sample(next_grid_ts)
            next_grid_ts += config.sampling_ms

        rebuild_book(
            action=str(record.get("action", "")),
            bids=record.get("bids", []),
            asks=record.get("asks", []),
            bid_book=bid_book,
            ask_book=ask_book,
        )
        current_state_ready = True

        # Track event-level mid for mid_std_2s (uses all OB event mids, not grid-sampled)
        event_mid = current_mid_from_book(bid_book, ask_book)
        if event_mid is not None:
            mid_window.append((event_ts, event_mid))

        while current_state_ready and next_grid_ts is not None and next_grid_ts == event_ts and next_grid_ts < day_end_ms:
            maybe_record_sample(next_grid_ts)
            next_grid_ts += config.sampling_ms

        if record_count % 1_000_000 == 0:
            log_progress(
                f"{date_str}: processed {record_count} order book records | sampled_rows={len(sample_rows)}"
            )

    while current_state_ready and next_grid_ts is not None and next_grid_ts < day_end_ms:
        maybe_record_sample(next_grid_ts)
        next_grid_ts += config.sampling_ms

    log_progress(
        f"{date_str}: finished sample-state build | order_book_records={record_count} | sampled_rows={len(sample_rows)}"
    )
    return sample_rows


def find_first_fill(
    trades: TradeData,
    start_idx: int,
    end_idx: int,
    side: str,
    quote_price: float,
    min_fill_size: float,
) -> Optional[int]:
    cumulative_size = 0.0
    expected_sign = -1 if side == "bid" else 1

    for idx in range(start_idx, end_idx):
        if trades.aggressor_signs[idx] != expected_sign:
            continue

        trade_price = trades.prices[idx]
        if side == "bid" and trade_price > quote_price:
            continue
        if side == "ask" and trade_price < quote_price:
            continue

        cumulative_size += trades.sizes[idx]
        if cumulative_size >= min_fill_size:
            return trades.timestamps[idx]

    return None


def build_fill_candidates(
    config: PipelineConfig,
    sample_rows: Sequence[SampleState],
    trades: TradeData,
) -> List[FillCandidate]:
    log_progress(f"building fill candidates for {len(sample_rows)} sampled states")
    candidates: List[FillCandidate] = []

    for index, sample in enumerate(sample_rows, start=1):
        if not sample.feature_window_ready:
            continue
        if sample.spread_bps <= 0:
            continue

        start_idx = bisect_right(trades.timestamps, sample.ts)
        end_idx = bisect_right(trades.timestamps, sample.ts + config.quote_active_ms)
        if start_idx >= end_idx:
            continue

        for offset_bps in config.candidate_offsets_bps:
            bid_quote = sample.mid * (1.0 - offset_bps / 10_000.0)
            bid_fill_ts = find_first_fill(trades, start_idx, end_idx, "bid", bid_quote, config.min_fill_size)
            if bid_fill_ts is not None:
                candidates.append(
                    FillCandidate(
                        sample_ts=sample.ts,
                        side="bid",
                        offset_bps=offset_bps,
                        fill_ts=bid_fill_ts,
                        entry_price=bid_quote,
                        entry_mid=sample.mid,
                        exit_target_ts=bid_fill_ts + config.exit_delay_ms,
                    )
                )

            ask_quote = sample.mid * (1.0 + offset_bps / 10_000.0)
            ask_fill_ts = find_first_fill(trades, start_idx, end_idx, "ask", ask_quote, config.min_fill_size)
            if ask_fill_ts is not None:
                candidates.append(
                    FillCandidate(
                        sample_ts=sample.ts,
                        side="ask",
                        offset_bps=offset_bps,
                        fill_ts=ask_fill_ts,
                        entry_price=ask_quote,
                        entry_mid=sample.mid,
                        exit_target_ts=ask_fill_ts + config.exit_delay_ms,
                    )
                )

        if index % 250 == 0:
            log_progress(
                f"fill candidate progress: processed_samples={index}/{len(sample_rows)} | candidates={len(candidates)}"
            )

    log_progress(f"finished fill candidate build | total_candidates={len(candidates)}")
    return candidates


def resolve_exit_prices(config: PipelineConfig, date_str: str, candidates: List[FillCandidate]) -> None:
    if not candidates:
        log_progress(f"{date_str}: no fill candidates to resolve exits for")
        return

    log_progress(f"{date_str}: resolving exits for {len(candidates)} fill candidates")
    pending = sorted(candidates, key=lambda candidate: candidate.exit_target_ts)
    pending_idx = 0

    for candidate_date in (date_str, next_date(date_str)):
        path = order_book_path(config, candidate_date)
        if not path.exists():
            continue

        log_progress(f"{date_str}: scanning exit mids from {path.name}")
        bid_book: Dict[float, float] = {}
        ask_book: Dict[float, float] = {}
        record_count = 0

        for record in iter_order_book_records(path, config.instrument):
            record_count += 1
            rebuild_book(
                action=str(record.get("action", "")),
                bids=record.get("bids", []),
                asks=record.get("asks", []),
                bid_book=bid_book,
                ask_book=ask_book,
            )

            current_mid = current_mid_from_book(bid_book, ask_book)
            if current_mid is None:
                continue

            event_ts = int(record["ts"])
            while pending_idx < len(pending) and pending[pending_idx].exit_target_ts <= event_ts:
                pending[pending_idx].exit_ts = event_ts
                pending[pending_idx].exit_mid = current_mid
                pending_idx += 1

                if pending_idx % 1_000 == 0:
                    log_progress(
                        f"{date_str}: resolved exits for {pending_idx}/{len(pending)} candidates"
                    )

            if pending_idx >= len(pending):
                log_progress(
                    f"{date_str}: finished exit resolution | resolved={pending_idx} | scanned_records={record_count}"
                )
                return

            if record_count % 1_000_000 == 0:
                log_progress(
                    f"{date_str}: exit scan progress on {path.name} | records={record_count} | resolved={pending_idx}/{len(pending)}"
                )

        log_progress(
            f"{date_str}: completed scan of {path.name} | scanned_records={record_count} | resolved={pending_idx}/{len(pending)}"
        )


def candidate_pnl_bps(config: PipelineConfig, candidate: FillCandidate) -> Optional[float]:
    if candidate.exit_mid is None:
        return None

    if candidate.side == "bid":
        return config.maker_rebate_bps + ((candidate.exit_mid - candidate.entry_price) / candidate.entry_mid) * 10_000.0
    return config.maker_rebate_bps + ((candidate.entry_price - candidate.exit_mid) / candidate.entry_mid) * 10_000.0


def choose_best_candidate(config: PipelineConfig, candidates: Sequence[FillCandidate]) -> Optional[Tuple[FillCandidate, float]]:
    best_candidate: Optional[FillCandidate] = None
    best_pnl: Optional[float] = None
    epsilon = 1e-12

    for candidate in candidates:
        pnl_bps = candidate_pnl_bps(config, candidate)
        if pnl_bps is None:
            continue

        if best_candidate is None or pnl_bps > best_pnl + epsilon:
            best_candidate = candidate
            best_pnl = pnl_bps
            continue

        if abs(pnl_bps - best_pnl) <= epsilon and candidate.offset_bps < best_candidate.offset_bps:
            best_candidate = candidate
            best_pnl = pnl_bps

    if best_candidate is None or best_pnl is None:
        return None
    return best_candidate, best_pnl


def labeled_schema():
    pa, _ = load_pyarrow()
    return pa.schema(
        [
            ("ts", pa.int64()),
            ("date", pa.string()),
            ("side", pa.string()),
            ("OBI5", pa.float64()),
            ("OBI25", pa.float64()),
            ("OBI400", pa.float64()),
            ("NTR_10s", pa.float64()),
            ("mid_std_2s", pa.float64()),
            ("spread_bps", pa.float64()),
            ("hour_of_day", pa.int32()),
            ("trade_flow_10s", pa.float64()),
            ("trade_count_10s", pa.int64()),
            ("cumulativeVolume_5bps", pa.float64()),
            ("optimal_offset_bps", pa.float64()),
            ("optimal_pnl_bps", pa.float64()),
            ("fill_ts", pa.int64()),
            ("exit_ts", pa.int64()),
            ("entry_price", pa.float64()),
            ("exit_mid", pa.float64()),
            ("valid_label_flag", pa.int8()),
        ]
    )


def build_labeled_rows(
    config: PipelineConfig,
    sample_rows: Sequence[SampleState],
    candidates: Sequence[FillCandidate],
) -> List[Dict[str, object]]:
    log_progress(
        f"building labeled rows from {len(sample_rows)} sampled states and {len(candidates)} fill candidates"
    )
    sample_by_ts = {sample.ts: sample for sample in sample_rows}
    grouped: Dict[Tuple[int, str], List[FillCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.sample_ts, candidate.side), []).append(candidate)

    labeled_rows: List[Dict[str, object]] = []
    for (sample_ts, side), side_candidates in sorted(grouped.items()):
        choice = choose_best_candidate(config, side_candidates)
        if choice is None:
            continue

        best_candidate, best_pnl_bps = choice
        sample = sample_by_ts[sample_ts]
        labeled_rows.append(
            {
                "ts": sample.ts,
                "date": sample.date,
                "side": side,
                "OBI5": sample.OBI5,
                "OBI25": sample.OBI25,
                "OBI400": sample.OBI400,
                "NTR_10s": sample.NTR_10s,
                "mid_std_2s": sample.mid_std_2s,
                "spread_bps": sample.spread_bps,
                "hour_of_day": sample.hour_of_day,
                "trade_flow_10s": sample.trade_flow_10s,
                "trade_count_10s": sample.trade_count_10s,
                "cumulativeVolume_5bps": sample.depth_5bps,
                "optimal_offset_bps": best_candidate.offset_bps,
                "optimal_pnl_bps": best_pnl_bps,
                "fill_ts": best_candidate.fill_ts,
                "exit_ts": best_candidate.exit_ts,
                "entry_price": best_candidate.entry_price,
                "exit_mid": best_candidate.exit_mid,
                "valid_label_flag": 1,
            }
        )

    log_progress(f"finished labeled row build | labeled_rows={len(labeled_rows)}")
    return labeled_rows


def write_parquet(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    pa, pq = load_pyarrow()
    ensure_dir(path.parent)
    row_list = list(rows)
    log_progress(f"writing parquet {path} | rows={len(row_list)}")
    table = pa.Table.from_pylist(row_list, schema=labeled_schema())
    pq.write_table(table, path)


def write_combined_dataset(output_root: Path, daily_paths: Sequence[Path]) -> Path:
    _, pq = load_pyarrow()
    combined_path = output_root / "labeled_dataset_btc_usdt_swap_side_separated.parquet"
    log_progress(f"combining {len(daily_paths)} daily parquet files into {combined_path}")
    writer = pq.ParquetWriter(combined_path, labeled_schema())
    try:
        for daily_path in daily_paths:
            log_progress(f"appending {daily_path.name} into combined parquet")
            writer.write_table(pq.read_table(daily_path))
    finally:
        writer.close()

    log_progress(f"finished combined parquet write: {combined_path}")
    return combined_path


def process_day(config: PipelineConfig, date_str: str, output_dirs: Dict[str, Path]) -> Tuple[Path, int]:
    log_progress(f"{date_str}: starting day-level processing")
    trades = load_trade_data_for_day(config, date_str)
    sample_rows = build_sample_states(config, date_str, trades)
    fill_candidates = build_fill_candidates(config, sample_rows, trades)
    resolve_exit_prices(config, date_str, fill_candidates)
    labeled_rows = build_labeled_rows(config, sample_rows, fill_candidates)

    daily_path = output_dirs["daily"] / f"labeled_side_rows_{date_str}.parquet"
    write_parquet(daily_path, labeled_rows)
    log_progress(
        f"{date_str}: finished day-level processing | sampled_rows={len(sample_rows)} | "
        f"fill_candidates={len(fill_candidates)} | labeled_rows={len(labeled_rows)}"
    )
    return daily_path, len(labeled_rows)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        order_book_dir=args.order_book_dir,
        trade_dir=args.trade_dir,
        output_root=args.output_root,
        sampling_ms=args.sampling_ms,
        samples_per_day=None if args.samples_per_day == 0 else args.samples_per_day,
        random_subsample=args.random_subsample,
        random_seed=args.random_seed,
        lookback_ms=args.lookback_ms,
        quote_active_ms=args.quote_active_ms,
        exit_delay_ms=args.exit_delay_ms,
        min_fill_size=args.min_fill_size,
    )
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if not config.order_book_dir.exists():
        raise FileNotFoundError(f"order_book_dir does not exist: {config.order_book_dir}")
    if not config.trade_dir.exists():
        raise FileNotFoundError(f"trade_dir does not exist: {config.trade_dir}")
    if config.sampling_ms <= 0:
        raise ValueError("sampling_ms must be positive")
    if config.lookback_ms <= 0:
        raise ValueError("lookback_ms must be positive")
    if config.mid_std_window_ms <= 0:
        raise ValueError("mid_std_window_ms must be positive")
    if config.quote_active_ms <= 0:
        raise ValueError("quote_active_ms must be positive")
    if config.exit_delay_ms <= 0:
        raise ValueError("exit_delay_ms must be positive")
    if config.lookback_ms % config.sampling_ms != 0:
        raise ValueError("lookback_ms must be an integer multiple of sampling_ms")
    if config.samples_per_day is not None and config.samples_per_day <= 0:
        raise ValueError("samples_per_day must be positive when provided")
    if config.min_fill_size <= 0:
        raise ValueError("min_fill_size must be positive")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dirs = init_output_dirs(config.output_root)

    dates = args.dates if args.dates else available_dates(config)
    if not dates:
        raise FileNotFoundError(f"No order book .data files found in {config.order_book_dir}")

    log_progress(
        f"starting run | dates={len(dates)} | sampling_ms={config.sampling_ms} | "
        f"samples_per_day={config.samples_per_day} | output_root={config.output_root}"
    )
    daily_paths: List[Path] = []
    total_rows = 0

    for date_str in dates:
        daily_path, row_count = process_day(config, date_str, output_dirs)
        daily_paths.append(daily_path)
        total_rows += row_count
        print(f"[step1] processed {date_str} | labeled_rows={row_count} | output={daily_path.name}")

    if not args.skip_combine and daily_paths:
        combined_path = write_combined_dataset(config.output_root, daily_paths)
        print(f"[step1] combined_dataset={combined_path} total_rows={total_rows}")

    log_progress(f"run complete | total_rows={total_rows} | days={len(daily_paths)}")


if __name__ == "__main__":
    main()
