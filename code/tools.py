

import json


"""
given a .data file, output an dataframe with x levels in the order book
the file will be in JSON format and the Top-level keys are ['instId', 'action', 'ts', 'asks', 'bids'], where 'asks' and 'bids' are lists of lists, each inner list has three elements: price, size and orders.
e.g.
Top-level keys: dict_keys(['instId', 'action', 'ts', 'asks', 'bids'])

[instId] type=str
  value: BTC-USDT-SWAP

[action] type=str
  value: snapshot

[ts] type=str
  value: 1767225600004

[asks] type=list
  list length: 400
keys of first item: N/A
  first item:  ['87593.1', '818.6', '35']

[bids] type=list
  list length: 400
keys of first item: N/A
  first item:  ['87593', '503.21', '31']

the output dataframe should have the following columns: ['instId', 'action', 'ts', 'ask_price_1', 'ask_size_1', 'ask_orders_1', 'bid_price_1', 'bid_size_1', 'bid_orders_1', ..., 'ask_price_x', 'ask_size_x', 'ask_orders_x', 'bid_price_x', 'bid_size_x', 'bid_orders_x']
"""
def parse_order_book(file_path, x):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("parse_order_book requires pandas to be installed") from exc

    if not isinstance(x, int) or x <= 0:
        raise ValueError("x must be a positive integer")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"{file_path} is empty")

    def _normalize_records(payload):
        if isinstance(payload, dict):
            if all(key in payload for key in ("instId", "action", "ts", "asks", "bids")):
                return [payload]
            if "data" in payload and isinstance(payload["data"], list):
                return [
                    item
                    for item in payload["data"]
                    if isinstance(item, dict)
                    and all(key in item for key in ("instId", "action", "ts", "asks", "bids"))
                ]
            raise ValueError("JSON object does not contain order book fields")

        if isinstance(payload, list):
            return [
                item
                for item in payload
                if isinstance(item, dict)
                and all(key in item for key in ("instId", "action", "ts", "asks", "bids"))
            ]

        raise ValueError("Unsupported JSON structure in file")

    try:
        records = _normalize_records(json.loads(content))
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            records.extend(_normalize_records(json.loads(line)))

    if not records:
        raise ValueError("No valid order book records found")

    def _get_level(side_levels, level_index):
        if level_index >= len(side_levels):
            return None, None, None

        level = side_levels[level_index]
        if not isinstance(level, (list, tuple)):
            return None, None, None

        price = level[0] if len(level) > 0 else None
        size = level[1] if len(level) > 1 else None
        orders = level[2] if len(level) > 2 else None
        return price, size, orders

    rows = []
    for record in records:
        row = {
            "instId": record.get("instId"),
            "action": record.get("action"),
            "ts": record.get("ts"),
        }

        asks = record.get("asks") or []
        bids = record.get("bids") or []

        for level in range(1, x + 1):
            ask_price, ask_size, ask_orders = _get_level(asks, level - 1)
            bid_price, bid_size, bid_orders = _get_level(bids, level - 1)

            row[f"ask_price_{level}"] = ask_price
            row[f"ask_size_{level}"] = ask_size
            row[f"ask_orders_{level}"] = ask_orders
            row[f"bid_price_{level}"] = bid_price
            row[f"bid_size_{level}"] = bid_size
            row[f"bid_orders_{level}"] = bid_orders

        rows.append(row)

    columns = ["instId", "action", "ts"]
    for level in range(1, x + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"ask_orders_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
                f"bid_orders_{level}",
            ]
        )

    return pd.DataFrame(rows, columns=columns)
