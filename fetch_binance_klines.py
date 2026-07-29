#!/usr/bin/env python3
"""
Fetch historical candlestick (kline) data from Binance.

Two modes:
  1. Bulk download (>= 1m intervals): Uses data.binance.vision pre-built ZIPs.
     Much faster for historical data — no rate limits.
  2. REST API (sub-minute intervals: 1s, 5s, 10s, 30s): Uses the Binance REST API
     with pagination (1000 candles per request, 100ms delay between requests).

Usage:
    # Default: BTCUSDT 1m, 5 years back (bulk download)
    python3 fetch_binance_klines.py

    # Custom interval and symbol
    python3 fetch_binance_klines.py --interval 5m --symbol ETHUSDT --years-back 2

    # Sub-minute: 1s data, last 1 day (REST API)
    python3 fetch_binance_klines.py --interval 1s --days-back 1

    # Sub-minute: 10s data, last 7 days
    python3 fetch_binance_klines.py --interval 10s --days-back 7
"""

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta

import requests

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"
SOURCE = "binance"
DEFAULT_YEARS_BACK = 5

BASE_URL = "https://data.binance.vision/data/spot"
REST_API_URL = "https://api.binance.com/api/v3/klines"

SUB_MINUTE_INTERVALS = {"1s", "5s", "10s", "30s"}

# Map interval strings to approximate millisecond durations (for pagination)
INTERVAL_MS = {
    "1s": 1_000,
    "5s": 5_000,
    "10s": 10_000,
    "30s": 30_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1D": 86_400_000,
}

CSV_HEADER = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def normalize_ts(val: str) -> str:
    """Normalize a timestamp to milliseconds. Binance switched from ms (13 digits)
    to us (16 digits) in 2025+ data files."""
    n = int(val)
    if n > 1e15:  # microseconds -> milliseconds
        return str(n // 1000)
    return val


def download_zip_csv(url: str) -> list[list]:
    """Download a ZIP file and extract CSV rows from it."""
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return []  # file doesn't exist yet (future month/day)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            rows = []
            for row in reader:
                # Normalize open_time (col 0) and close_time (col 6) to milliseconds
                row[0] = normalize_ts(row[0])
                row[6] = normalize_ts(row[6])
                rows.append(row)
            return rows


def generate_monthly_urls(symbol: str, interval: str, start: datetime, end: datetime):
    """Generate monthly kline ZIP URLs for the given date range."""
    current = start.replace(day=1)
    while current <= end:
        year_month = current.strftime("%Y-%m")
        url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"
        yield current, url
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def generate_daily_urls(symbol: str, interval: str, start_date: datetime, end_date: datetime):
    """Generate daily kline ZIP URLs for the given date range."""
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        yield current, url
        current += timedelta(days=1)


# ─── REST API fetcher (sub-minute) ──────────────────────────────────────────

def fetch_rest_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    delay: float = 0.1,
) -> list[list]:
    """Fetch klines from Binance REST API with pagination.

    Returns rows in the same format as the bulk CSV: each row is a list of strings.
    Binance returns max 1000 candles per request.
    """
    all_rows = []
    current_start = start_ms
    request_count = 0

    interval_ms = INTERVAL_MS.get(interval, 60_000)
    total_candles_est = (end_ms - start_ms) // interval_ms
    total_requests_est = max(1, total_candles_est // limit)

    print(f"  Estimated: ~{total_candles_est:,} candles, ~{total_requests_est:,} requests")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        resp = requests.get(REST_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for candle in data:
            # Binance REST returns: [open_time, open, high, low, close, volume,
            #   close_time, quote_volume, trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
            row = [
                str(candle[0]),   # open_time
                str(candle[1]),   # open
                str(candle[2]),   # high
                str(candle[3]),   # low
                str(candle[4]),   # close
                str(candle[5]),   # volume
                str(candle[6]),   # close_time
                str(candle[7]),   # quote_volume
                str(candle[8]),   # trades
                str(candle[9]),   # taker_buy_base_volume
                str(candle[10]),  # taker_buy_quote_volume
            ]
            all_rows.append(row)

        request_count += 1
        last_close_time = int(data[-1][6])
        current_start = last_close_time + 1

        if request_count % 50 == 0:
            print(f"  [{request_count} requests] {len(all_rows):,} candles so far ...")

        # Rate limiting
        if len(data) == limit:
            time.sleep(delay)
        else:
            break  # Got fewer than limit, we're done

    print(f"  Completed: {request_count} requests, {len(all_rows):,} candles")
    return all_rows


# ─── Bulk download (>= 1m) ──────────────────────────────────────────────────

def fetch_bulk(symbol: str, interval: str, start: datetime, now: datetime) -> list[list]:
    """Fetch klines via data.binance.vision bulk monthly/daily ZIPs."""
    all_rows = []

    # Monthly data is typically available up to 2 months ago.
    # Use daily files to fill the gap to today.
    monthly_urls = list(generate_monthly_urls(symbol, interval, start, now))
    print(f"Phase 1: Downloading {len(monthly_urls)} monthly files ...")

    last_monthly_date = start
    for i, (date, url) in enumerate(monthly_urls, 1):
        label = date.strftime("%Y-%m")
        try:
            rows = download_zip_csv(url)
            if rows:
                all_rows.extend(rows)
                last_monthly_date = date
                print(f"  [{i:3d}/{len(monthly_urls)}]  {label}  ->  {len(rows):>8,} candles")
            else:
                print(f"  [{i:3d}/{len(monthly_urls)}]  {label}  ->  not available (switching to daily)")
                break
        except Exception as e:
            print(f"  [{i:3d}/{len(monthly_urls)}]  {label}  ->  error: {e} (switching to daily)")
            break
    else:
        # All monthly files downloaded — no daily fill needed
        last_monthly_date = None

    # Daily ZIPs for the remaining gap
    if last_monthly_date is not None:
        daily_start = last_monthly_date.replace(day=1)
        daily_urls = list(generate_daily_urls(symbol, interval, daily_start, now))
        if daily_urls:
            print(f"\nPhase 2: Downloading {len(daily_urls)} daily files ({daily_start.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d')}) ...")

            for i, (date, url) in enumerate(daily_urls, 1):
                label = date.strftime("%Y-%m-%d")
                try:
                    rows = download_zip_csv(url)
                    if rows:
                        all_rows.extend(rows)
                        if i % 10 == 0 or i == len(daily_urls):
                            print(f"  [{i:3d}/{len(daily_urls)}]  {label}  ->  {len(rows):>6,} candles")
                except Exception as e:
                    print(f"  [{i:3d}/{len(daily_urls)}]  {label}  ->  error: {e}")

    return all_rows


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance kline data (bulk download or REST API for sub-minute)"
    )
    parser.add_argument("--interval", "-i", default=DEFAULT_INTERVAL,
                        help=f"Candle interval (default: {DEFAULT_INTERVAL}). "
                             f"Sub-minute: {', '.join(sorted(SUB_MINUTE_INTERVALS))} use REST API.")
    parser.add_argument("--symbol", "-s", default=DEFAULT_SYMBOL,
                        help=f"Trading pair (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--years-back", type=int, default=None,
                        help=f"Years of history to fetch (default: {DEFAULT_YEARS_BACK} for bulk)")
    parser.add_argument("--days-back", type=int, default=None,
                        help="Days of history to fetch (overrides --years-back)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/)")

    args = parser.parse_args()

    symbol = args.symbol
    interval = args.interval

    data_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(data_dir, f"{SOURCE}_{symbol}_{interval}_klines.csv")

    now = datetime.now(timezone.utc)

    # Determine time range
    if args.days_back is not None:
        start = now - timedelta(days=args.days_back)
    elif args.years_back is not None:
        start = datetime(now.year - args.years_back, now.month, now.day, tzinfo=timezone.utc)
    elif interval in SUB_MINUTE_INTERVALS:
        # Default to 1 day for sub-minute
        start = now - timedelta(days=1)
    else:
        start = datetime(now.year - DEFAULT_YEARS_BACK, now.month, now.day, tzinfo=timezone.utc)

    is_sub_minute = interval in SUB_MINUTE_INTERVALS

    print(f"Symbol:      {symbol}")
    print(f"Interval:    {interval}")
    print(f"Mode:        {'REST API (sub-minute)' if is_sub_minute else 'Bulk download (data.binance.vision)'}")
    print(f"Range:       {start.strftime('%Y-%m-%d %H:%M')} -> {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Output:      {output_file}")
    print()

    # Fetch data
    if is_sub_minute:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        all_rows = fetch_rest_klines(symbol, interval, start_ms, end_ms)
    else:
        all_rows = fetch_bulk(symbol, interval, start, now)

    # Sort, filter, deduplicate
    print(f"\nSorting {len(all_rows):,} rows by open_time ...")

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    all_rows = [r for r in all_rows if start_ms <= int(r[0]) <= end_ms]

    all_rows.sort(key=lambda r: int(r[0]))

    seen = set()
    deduped = []
    for row in all_rows:
        ts = int(row[0])
        if ts not in seen:
            seen.add(ts)
            deduped.append(row)
    all_rows = deduped

    print(f"After dedup: {len(all_rows):,} rows")

    if not all_rows:
        print("\nNo data fetched. Check your interval and date range.")
        sys.exit(1)

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in all_rows:
            writer.writerow([
                int(row[0]),       # open_time
                row[1],            # open
                row[2],            # high
                row[3],            # low
                row[4],            # close
                row[5],            # volume
                int(row[6]),       # close_time
                row[7],            # quote_volume
                int(row[8]),       # trades
                row[9],            # taker_buy_base_volume
                row[10],           # taker_buy_quote_volume
            ])

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    first_ts = ms_to_iso(int(all_rows[0][0]))
    last_ts = ms_to_iso(int(all_rows[-1][0]))

    print(f"\nDone!")
    print(f"  Rows:       {len(all_rows):,}")
    print(f"  First:      {first_ts}")
    print(f"  Last:       {last_ts}")
    print(f"  File size:  {file_size_mb:.1f} MB")
    print(f"  Output:     {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
