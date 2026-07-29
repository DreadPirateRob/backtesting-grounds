#!/usr/bin/env python3
"""
Fetch funding rate and open interest data from Binance Futures REST API.

Funding Rate API: GET https://fapi.binance.com/fapi/v1/fundingRate
Open Interest History API: GET https://fapi.binance.com/futures/data/openInterestHist

Usage:
    # Default: BTCUSDT, 3 years funding + 1 year OI
    python3 fetch_funding_oi.py

    # Custom symbol and range
    python3 fetch_funding_oi.py --symbol ETHUSDT --days-back 180

    # Funding only
    python3 fetch_funding_oi.py --funding-only

    # OI only, custom period
    python3 fetch_funding_oi.py --oi-only --oi-period 1h
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_FUNDING_DAYS = 365 * 3
DEFAULT_OI_DAYS = 365
DEFAULT_OI_PERIOD = "5m"

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"

FUNDING_LIMIT = 1000
OI_LIMIT = 500
REQUEST_DELAY = 0.1
RATE_LIMIT_DELAY = 60

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def request_with_retry(url: str, params: dict, max_retries: int = 5) -> list[dict]:
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = RATE_LIMIT_DELAY * (attempt + 1)
            print(f"  Rate limited (429). Waiting {wait}s ...")
            time.sleep(wait)
            continue
        if resp.status_code == 418:
            wait = RATE_LIMIT_DELAY * 5
            print(f"  IP ban (418). Waiting {wait}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Max retries ({max_retries}) exceeded for {url}")


# ─── Funding Rate Fetcher ────────────────────────────────────────────────────

def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    all_records: list[dict] = []
    current_start = start_ms
    request_count = 0

    print(f"Fetching funding rates for {symbol} ...")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": FUNDING_LIMIT,
        }

        data = request_with_retry(FUNDING_URL, params)
        if not data:
            break

        all_records.extend(data)
        request_count += 1

        last_time = int(data[-1]["fundingTime"])
        current_start = last_time + 1

        if request_count % 10 == 0:
            print(f"  [{request_count} requests] {len(all_records):,} records, "
                  f"latest: {ms_to_iso(last_time)}")

        if len(data) < FUNDING_LIMIT:
            break

        time.sleep(REQUEST_DELAY)

    print(f"  Completed: {request_count} requests, {len(all_records):,} records")
    return all_records


def save_funding_rates(records: list[dict], output_path: str) -> int:
    seen: set[int] = set()
    deduped: list[dict] = []
    for r in records:
        ts = int(r["fundingTime"])
        if ts not in seen:
            seen.add(ts)
            deduped.append(r)
    deduped.sort(key=lambda r: int(r["fundingTime"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "funding_rate", "mark_price"])
        for r in deduped:
            writer.writerow([
                int(r["fundingTime"]),
                r["fundingRate"],
                r.get("markPrice", ""),
            ])

    return len(deduped)


# ─── Open Interest Fetcher ───────────────────────────────────────────────────

def fetch_open_interest(symbol: str, period: str, start_ms: int, end_ms: int) -> list[dict]:
    all_records: list[dict] = []
    current_start = start_ms
    request_count = 0

    print(f"Fetching open interest for {symbol} (period={period}) ...")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": OI_LIMIT,
        }

        data = request_with_retry(OI_HIST_URL, params)
        if not data:
            break

        all_records.extend(data)
        request_count += 1

        last_time = int(data[-1]["timestamp"])
        current_start = last_time + 1

        if request_count % 10 == 0:
            print(f"  [{request_count} requests] {len(all_records):,} records, "
                  f"latest: {ms_to_iso(last_time)}")

        if len(data) < OI_LIMIT:
            break

        time.sleep(REQUEST_DELAY)

    print(f"  Completed: {request_count} requests, {len(all_records):,} records")
    return all_records


def save_open_interest(records: list[dict], output_path: str) -> int:
    seen: set[int] = set()
    deduped: list[dict] = []
    for r in records:
        ts = int(r["timestamp"])
        if ts not in seen:
            seen.add(ts)
            deduped.append(r)
    deduped.sort(key=lambda r: int(r["timestamp"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open_interest", "open_interest_value"])
        for r in deduped:
            writer.writerow([
                int(r["timestamp"]),
                r["sumOpenInterest"],
                r["sumOpenInterestValue"],
            ])

    return len(deduped)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance Futures funding rate and open interest data"
    )
    parser.add_argument("--symbol", "-s", default=DEFAULT_SYMBOL,
                        help=f"Trading pair (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Days of history (overrides defaults for both)")
    parser.add_argument("--funding-days", type=int, default=DEFAULT_FUNDING_DAYS,
                        help=f"Days of funding rate history (default: {DEFAULT_FUNDING_DAYS})")
    parser.add_argument("--oi-days", type=int, default=DEFAULT_OI_DAYS,
                        help=f"Days of OI history (default: {DEFAULT_OI_DAYS})")
    parser.add_argument("--oi-period", default=DEFAULT_OI_PERIOD,
                        choices=["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
                        help=f"OI granularity (default: {DEFAULT_OI_PERIOD})")
    parser.add_argument("--funding-only", action="store_true",
                        help="Only fetch funding rates")
    parser.add_argument("--oi-only", action="store_true",
                        help="Only fetch open interest")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/)")

    args = parser.parse_args()

    symbol = args.symbol
    data_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    now = datetime.now(timezone.utc)

    if args.days_back is not None:
        funding_days = args.days_back
        oi_days = args.days_back
    else:
        funding_days = args.funding_days
        oi_days = args.oi_days

    fetch_funding = not args.oi_only
    fetch_oi = not args.funding_only

    print(f"Symbol:         {symbol}")
    if fetch_funding:
        print(f"Funding range:  {funding_days} days back")
    if fetch_oi:
        print(f"OI range:       {oi_days} days back (period={args.oi_period})")
    print()

    # Funding rates
    if fetch_funding:
        funding_start = now - timedelta(days=funding_days)
        funding_start_ms = int(funding_start.timestamp() * 1000)
        funding_end_ms = int(now.timestamp() * 1000)

        records = fetch_funding_rates(symbol, funding_start_ms, funding_end_ms)
        if records:
            output_path = os.path.join(data_dir, f"binance_{symbol}_funding_rates.csv")
            count = save_funding_rates(records, output_path)
            first_ts = ms_to_iso(int(records[0]["fundingTime"]))
            last_ts = ms_to_iso(int(records[-1]["fundingTime"]))
            file_size = os.path.getsize(output_path) / 1024
            print(f"\nFunding rates saved:")
            print(f"  Rows:   {count:,}")
            print(f"  First:  {first_ts}")
            print(f"  Last:   {last_ts}")
            print(f"  Size:   {file_size:.1f} KB")
            print(f"  Path:   {output_path}")
        else:
            print("\nNo funding rate data returned.")
        print()

    # Open interest
    if fetch_oi:
        oi_start = now - timedelta(days=oi_days)
        oi_start_ms = int(oi_start.timestamp() * 1000)
        oi_end_ms = int(now.timestamp() * 1000)

        records = fetch_open_interest(symbol, args.oi_period, oi_start_ms, oi_end_ms)
        if records:
            output_path = os.path.join(data_dir, f"binance_{symbol}_open_interest.csv")
            count = save_open_interest(records, output_path)
            first_ts = ms_to_iso(int(records[0]["timestamp"]))
            last_ts = ms_to_iso(int(records[-1]["timestamp"]))
            file_size = os.path.getsize(output_path) / 1024
            print(f"\nOpen interest saved:")
            print(f"  Rows:   {count:,}")
            print(f"  First:  {first_ts}")
            print(f"  Last:   {last_ts}")
            print(f"  Size:   {file_size:.1f} KB")
            print(f"  Path:   {output_path}")
        else:
            print("\nNo open interest data returned.")

    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
