from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import aiohttp
import dotenv
import pandas as pd

from services.replay_utils import bucket_to_seconds, event_map, process_intervals

dotenv.load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    raise ValueError("API_BASE_URL environment variable is required")
API_BASE_URL = API_BASE_URL.rstrip("/")

DATA_ROOT = Path("data")


@dataclass
class ReplayState:
    track_flag: Optional[str] = None
    driver_flag: Dict[int, str] = field(default_factory=dict)

    pit_windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = field(default_factory=dict)
    slow_windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = field(default_factory=dict)
    finished_drivers: Set[int] = field(default_factory=set)
    finished_at: Dict[int, pd.Timestamp] = field(default_factory=dict)
    finished_position: Dict[int, int] = field(default_factory=dict)

    last_position: Dict[int, int] = field(default_factory=dict)


def _cache_path(session_key: str, name: str) -> Path:
    return DATA_ROOT / str(session_key) / name


async def fetch_json(
    session: aiohttp.ClientSession,
    path: str,
    params: dict,
    *,
    cache: bool = True,
    cache_name: Optional[str] = None,
    retries: int = 3,
) -> List[dict]:
    cache_file = _cache_path(str(params.get("session_key", "")), cache_name) if cache and cache_name else None
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())

    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params) as response:
                last_error = str(response.status)
                if response.status == 200:
                    payload = await response.json()
                    if cache_file:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps(payload, indent=2))
                    return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        await asyncio.sleep(0.5 * attempt)

    print(f"Failed to fetch {url} after {retries} attempts (last status {last_error})")
    return []

async def fetch_json_openf1(
    session: aiohttp.ClientSession,
    path: str,
    params: dict,
    *,
    cache: bool = True,
    cache_name: Optional[str] = None,
    retries: int = 2,
) -> List[dict]:
    """
    Fallback fetch that targets the public OpenF1 API directly.
    """
    cache_file = _cache_path(str(params.get("session_key", "")), cache_name) if cache and cache_name else None
    if cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached:
            return cached

    url = f"https://api.openf1.org/v1/{path.lstrip('/')}"
    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params) as response:
                print(f"Fetching OpenF1 API: {url} with params {params} (attempt {attempt})")

                last_error = str(response.status)
                if response.status == 200:
                    payload = await response.json()
                    if cache_file:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps(payload, indent=2))
                    print(payload)
                    return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        await asyncio.sleep(0.25 * attempt)

    print(f"Failed to fetch {url} after {retries} attempts (last status {last_error})")
    return []

async def get_laps(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "laps",
        {"session_key": session_key},
        cache=cache,
        cache_name="laps.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # parse time if present
    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], utc=True, errors="coerce")

    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["lap_number", "lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["st_speed", "i1_speed", "i2_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "is_pit_out_lap" in df.columns:
        df["is_pit_out_lap"] = df["is_pit_out_lap"].astype("boolean")

    return df


async def get_race_control(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "race_control",
        {"session_key": session_key},
        cache=cache,
        cache_name="race_control.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    return df


async def get_pit(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "pit",
        {"session_key": session_key},
        cache=cache,
        cache_name="pit.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    df["pit_duration"] = pd.to_numeric(df.get("pit_duration"), errors="coerce")
    return df


async def get_position(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "position",
        {"session_key": session_key},
        cache=cache,
        cache_name="position.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df.get("position"), errors="coerce").astype("Int64")
    return df


async def get_overtakes(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "overtakes",
        {"session_key": session_key},
        cache=cache,
        cache_name="overtakes.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = bucket_to_seconds(df, "date")
    if "overtaking_driver_number" in df.columns and "driver_number" not in df.columns:
        df["driver_number"] = df["overtaking_driver_number"]
    for col in ("driver_number", "overtaken_driver_number", "victim"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


async def get_session_intervals(
    session: aiohttp.ClientSession,
    session_key: str,
    *,
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = await fetch_json(
        session,
        "intervals",
        {"session_key": session_key},
        cache=cache,
        cache_name="intervals.json",
    )
    df = pd.DataFrame(data)
    return process_intervals(df)

async def get_car_data(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "car_data",
        {"session_key": session_key},
        cache=cache,
        cache_name="car_data.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["speed", "throttle", "brake", "rpm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "n_gear" in df.columns:
        df["n_gear"] = pd.to_numeric(df["n_gear"], errors="coerce").astype("Int64")

    if "drs" in df.columns:
        df["drs"] = pd.to_numeric(df["drs"], errors="coerce").astype("Int64")

    return df


async def get_stints(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "stints",
        {"session_key": session_key},
        cache=cache,
        cache_name="stints.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["lap_start", "lap_end", "stint_number", "tyre_age_at_start"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "tyre_compound" in df.columns:
        df["tyre_compound"] = df["tyre_compound"].astype("string")

    return df


async def get_team_radio(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "team_radio",
        {"session_key": session_key},
        cache=cache,
        cache_name="team_radio.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    return df


async def get_weather(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "weather",
        {"session_key": session_key},
        cache=cache,
        cache_name="weather.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = bucket_to_seconds(df, "date")
    numeric_cols = [
        "air_temperature",
        "humidity",
        "pressure",
        "rainfall",
        "track_temperature",
        "wind_direction",
        "wind_speed",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


async def get_location(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    data = await fetch_json(
        session,
        "location",
        {"session_key": session_key},
        cache=cache,
        cache_name="location.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["x", "y", "z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


async def get_starting_grid(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    # Always prefer the public OpenF1 API and cache locally
    data = await fetch_json_openf1(
        session,
        "starting_grid",
        {"session_key": session_key},
        cache=cache,
        cache_name="starting_grid.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["position", "grid_position"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


async def get_session_result(session: aiohttp.ClientSession, session_key: str, *, cache: bool = True) -> pd.DataFrame:
    # Always prefer the public OpenF1 API and cache locally
    data = await fetch_json_openf1(
        session,
        "session_result",
        {"session_key": session_key},
        cache=cache,
        cache_name="session_result.json",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    for c in ["position", "starting_position", "points", "laps", "interval", "time_retired"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def apply_events(state: ReplayState, t: pd.Timestamp, events: List[dict]) -> None:
    for event in events:
        kind = event.get("kind")

        if kind == "race_control":
            if event.get("scope") == "Track" and event.get("category") == "Flag":
                state.track_flag = event.get("flag")
            if event.get("scope") == "Driver" and event.get("category") == "Flag":
                dn = event.get("driver_number")
                if pd.notna(dn):
                    state.driver_flag[int(dn)] = event.get("flag")

        elif kind == "position":
            dn = event.get("driver_number")
            pos = event.get("position")
            if pd.notna(dn) and pd.notna(pos):
                state.last_position[int(dn)] = int(pos)


def _merge_windows(
    windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    merged: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for dn, ws in windows.items():
        ws = sorted(ws, key=lambda x: x[0])
        out: List[list[pd.Timestamp]] = []
        for s, e in ws:
            if not out or s > out[-1][1]:
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        merged[dn] = [(s, e) for s, e in out]
    return merged


def build_pit_windows(
    pit_df: pd.DataFrame,
    *,
    pre_buffer_s: float = 2.0,
    post_buffer_s: float = 25.0,
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    if pit_df is None or pit_df.empty:
        return {}

    windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)

    for r in pit_df.itertuples(index=False):
        dn = getattr(r, "driver_number", None)
        dur = getattr(r, "pit_duration", None)
        t0 = getattr(r, "t", None)

        if pd.isna(dn) or pd.isna(dur) or pd.isna(t0):
            continue

        dn = int(dn)
        pit_t = pd.Timestamp(t0).floor("1s")
        dur_s = float(dur)

        start = pit_t - pd.to_timedelta(dur_s + pre_buffer_s, unit="s")
        end = pit_t + pd.to_timedelta(post_buffer_s, unit="s")
        windows[dn].append((start, end))

    return _merge_windows(windows)

def build_slow_windows_from_intervals(
    driver_t: pd.DataFrame,
    synthetic_interval_long: pd.DataFrame,
    *,
    interval_jump_s_per_s: float = 2.0,  # losing >= 2.0s to car-ahead in 1s
    min_run_s: int = 5,                  # must persist for >= 5 seconds
    pad_before_s: int = 1,
    pad_after_s: int = 3,
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Detect "slow/anomaly" windows using ONLY interval-to-car-ahead (interval_synth_s).

    Why: gap_to_leader_s can jump for the whole field (leader changes / timing resets),
    which causes massive false positives. interval-to-ahead is more stable.

    Slow condition: d(interval_synth_s) >= interval_jump_s_per_s for >= min_run_s seconds.
    """
    if (
        synthetic_interval_long is None
        or synthetic_interval_long.empty
        or "interval_synth_s" not in synthetic_interval_long.columns
    ):
        return {}

    s = synthetic_interval_long.dropna(subset=["t", "driver_number"]).copy()
    s["driver_number"] = s["driver_number"].astype(int)

    # driver x time matrix of interval to car ahead
    int_pivot = s.pivot(index="t", columns="driver_number", values="interval_synth_s").sort_index()

    slow: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)

    for dn in int_pivot.columns:
        ser = int_pivot[dn]

        # per-second change in interval (positive = losing time to car ahead)
        d_int = ser.diff()

        # only care about positive growth; ignore improvements (negative)
        d_int = d_int.clip(lower=0)

        slow_mask = (d_int >= interval_jump_s_per_s).fillna(False)
        idx = slow_mask.index

        run_start: Optional[pd.Timestamp] = None
        run_len = 0

        for i, is_slow in enumerate(slow_mask.values):
            if is_slow:
                if run_start is None:
                    run_start = idx[i]
                    run_len = 1
                else:
                    run_len += 1
            else:
                if run_start is not None and run_len >= min_run_s:
                    start = run_start - pd.to_timedelta(pad_before_s, unit="s")
                    end = idx[i - 1] + pd.to_timedelta(pad_after_s, unit="s")
                    slow[int(dn)].append((start, end))
                run_start = None
                run_len = 0

        # tail
        if run_start is not None and run_len >= min_run_s:
            start = run_start - pd.to_timedelta(pad_before_s, unit="s")
            end = idx[-1] + pd.to_timedelta(pad_after_s, unit="s")
            slow[int(dn)].append((start, end))

    return _merge_windows(slow)

def build_speed_per_second(car_df: pd.DataFrame, *, agg: str = "mean") -> pd.Series:
    """
    Returns a Series indexed by (t, driver_number) -> speed_kmh
    agg: 'mean' | 'last' | 'max'
    """
    if car_df is None or car_df.empty:
        return pd.Series(dtype="float64")

    x = car_df.dropna(subset=["t", "driver_number", "speed"]).copy()
    x["driver_number"] = x["driver_number"].astype(int)

    if agg == "last":
        x = x.sort_values("date")
        out = x.groupby(["t", "driver_number"])["speed"].last()
    elif agg == "max":
        out = x.groupby(["t", "driver_number"])["speed"].max()
    else:
        out = x.groupby(["t", "driver_number"])["speed"].mean()

    return out


def build_slow_windows_hybrid(
    driver_t: pd.DataFrame,
    synthetic_interval_long: pd.DataFrame,
    pos_df: pd.DataFrame,
    *,
    interval_jump_s_per_s: float = 2.0,   # for non-leaders: losing to car ahead quickly
    leader_catch_s_per_s: float = 1.5,    # for leader: P2 is catching at >= 1.5s/s
    min_run_s: int = 3,
    pad_before_s: int = 1,
    pad_after_s: int = 3,
) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Slow windows:
    - Non-leaders: interval_synth_s increasing quickly (losing time to car ahead)
    - Leader: P2 gap_to_leader shrinking quickly (leader is slow)
    """
    windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)

    # --- (A) Non-leader slow via interval-to-ahead ---
    if (
        synthetic_interval_long is not None
        and not synthetic_interval_long.empty
        and "interval_synth_s" in synthetic_interval_long.columns
    ):
        s = synthetic_interval_long.dropna(subset=["t", "driver_number"]).copy()
        s["driver_number"] = s["driver_number"].astype(int)
        int_pivot = s.pivot(index="t", columns="driver_number", values="interval_synth_s").sort_index()

        for dn in int_pivot.columns:
            d_int = int_pivot[dn].diff().clip(lower=0)
            slow_mask = (d_int >= interval_jump_s_per_s).fillna(False)
            idx = slow_mask.index

            run_start = None
            run_len = 0
            for i, is_slow in enumerate(slow_mask.values):
                if is_slow:
                    if run_start is None:
                        run_start = idx[i]
                        run_len = 1
                    else:
                        run_len += 1
                else:
                    if run_start is not None and run_len >= min_run_s:
                        start = run_start - pd.to_timedelta(pad_before_s, unit="s")
                        end = idx[i - 1] + pd.to_timedelta(pad_after_s, unit="s")
                        windows[int(dn)].append((start, end))
                    run_start = None
                    run_len = 0

            if run_start is not None and run_len >= min_run_s:
                start = run_start - pd.to_timedelta(pad_before_s, unit="s")
                end = idx[-1] + pd.to_timedelta(pad_after_s, unit="s")
                windows[int(dn)].append((start, end))

    # --- (B) Leader slow via P2 catching quickly ---
    if (
        driver_t is not None and not driver_t.empty
        and pos_df is not None and not pos_df.empty
        and "gap_to_leader_s" in driver_t.columns
    ):
        # leader at each second (from position feed)
        p = pos_df.dropna(subset=["t", "driver_number", "position"]).copy()
        p["driver_number"] = p["driver_number"].astype(int)
        p["position"] = p["position"].astype(int)
        leader_by_t = (
            p[p["position"] == 1]
            .set_index("t")["driver_number"]
            .to_dict()
        )

        # P2 gap to leader at each second
        g = driver_t.dropna(subset=["t", "driver_number"]).copy()
        g["driver_number"] = g["driver_number"].astype(int)
        gap_pivot = g.pivot(index="t", columns="driver_number", values="gap_to_leader_s").sort_index()

        # find P2 each second using position feed
        p2_by_t = (
            p[p["position"] == 2]
            .set_index("t")["driver_number"]
            .to_dict()
        )

        # Build a “leader slow mask” per leader by checking P2 catch rate
        # catch_rate = -diff(P2_gap_to_leader); positive means P2 is gaining
        # If catch_rate >= leader_catch_s_per_s for min_run_s => leader is slow
        times = list(gap_pivot.index)
        # group by leader driver: collect seconds where leader is slow
        leader_slow_seconds: Dict[int, List[pd.Timestamp]] = defaultdict(list)

        for i in range(1, len(times)):
            t = times[i]
            t_prev = times[i - 1]

            leader = leader_by_t.get(t)
            p2 = p2_by_t.get(t)
            if leader is None or p2 is None:
                continue
            if p2 not in gap_pivot.columns:
                continue

            gap_now = gap_pivot.at[t, p2]
            gap_prev = gap_pivot.at[t_prev, p2]
            if pd.isna(gap_now) or pd.isna(gap_prev):
                continue

            catch_rate = float(gap_prev - gap_now)  # >0 => catching
            if catch_rate >= leader_catch_s_per_s:
                leader_slow_seconds[int(leader)].append(t)

        # Convert seconds list into windows per leader with run-length + padding
        for leader, secs in leader_slow_seconds.items():
            if not secs:
                continue
            secs = sorted(secs)
            run_start = secs[0]
            run_end = secs[0]
            run_len = 1

            def flush_run(s, e, n):
                if n >= min_run_s:
                    windows[leader].append(
                        (
                            s - pd.to_timedelta(pad_before_s, unit="s"),
                            e + pd.to_timedelta(pad_after_s, unit="s"),
                        )
                    )

            for tt in secs[1:]:
                if (tt - run_end) <= pd.to_timedelta(1, unit="s"):
                    run_end = tt
                    run_len += 1
                else:
                    flush_run(run_start, run_end, run_len)
                    run_start = run_end = tt
                    run_len = 1
            flush_run(run_start, run_end, run_len)

    return _merge_windows(windows)


def _build_window_map_seconds(
    windows: Dict[Tuple[int, int], List[dict]],
    *,
    per_second: bool = True,
    min_gain_s: float = 0.0,
    min_gain_rate_s_per_s: float = 0.0,
    include_payload: bool = False,
) -> Dict[pd.Timestamp, List[Any]]:
    """
    Expand pair windows into a lookup for replay annotation.

    per_second=True => include every second inside the window.
    per_second=False => only include the window start timestamp.
    """
    per_t: Dict[pd.Timestamp, List[Tuple[int, int]]] = defaultdict(list)
    for pair, ws in windows.items():
        for w in ws:
            start = w.get("start")
            end = w.get("end")
            if pd.isna(start) or pd.isna(end):
                continue
            tg = float(w.get("total_gain_s", 0) or 0)
            gr = float(w.get("avg_gain_rate_s_per_s", 0) or 0)
            if tg < min_gain_s or gr < min_gain_rate_s_per_s:
                continue
            payload = {
                "pair": pair,
                "start_gap_s": w.get("start_gap_s"),
                "end_gap_s": w.get("end_gap_s"),
                "total_gain_s": tg,
            } if include_payload else pair
            if per_second:
                for t in pd.date_range(start, end, freq="1s"):
                    per_t[t].append(payload)
            else:
                per_t[start].append(payload)
    return per_t


def is_in_windows(
    windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    dn: int,
    t: pd.Timestamp,
) -> bool:
    for s, e in windows.get(int(dn), []):
        if s <= t <= e:
            return True
    return False


def is_in_pit(state: ReplayState, dn: int, t: pd.Timestamp) -> bool:
    return is_in_windows(state.pit_windows, dn, t)


def is_in_pit_buffered(state: ReplayState, dn: int, t: pd.Timestamp, *, linger_after_s: float = 3.0) -> bool:
    """
    Stronger pit filter: true during pit window AND for linger_after_s seconds after exit.
    """
    for s, e in state.pit_windows.get(int(dn), []):
        if s <= t <= e:
            return True
        if t > e and (t - e) <= pd.to_timedelta(linger_after_s, unit="s"):
            return True
    return False


def is_slow(state: ReplayState, dn: int, t: pd.Timestamp) -> bool:
    return is_in_windows(state.slow_windows, dn, t)


def is_bad_context(state: ReplayState, dn: int, t: pd.Timestamp) -> bool:
    return is_in_pit_buffered(state, dn, t) or is_slow(state, dn, t)


def build_finished_drivers(session_result_df: pd.DataFrame) -> Set[int]:
    """
    Determine which drivers finished the race using session_result feed.
    """
    if session_result_df is None or session_result_df.empty:
        return set()

    df = session_result_df.copy()
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")

    finished_mask = pd.Series(False, index=df.index)

    if "status" in df.columns:
        finished_mask |= df["status"].astype(str).str.lower().str.contains("finish|chequer")

    if "time_retired" in df.columns:
        finished_mask |= df["time_retired"].isna()

    if "position" in df.columns:
        finished_mask |= df["position"].notna()

    drivers = df.loc[finished_mask, "driver_number"].dropna().astype(int)
    return set(drivers.tolist())


def build_finished_positions(session_result_df: pd.DataFrame) -> Dict[int, int]:
    """
    Extract classified finishing positions by driver.
    """
    if session_result_df is None or session_result_df.empty:
        return {}
    df = session_result_df.copy()
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df.get("position"), errors="coerce").astype("Int64")
    df = df.dropna(subset=["driver_number", "position"])
    return {int(r.driver_number): int(r.position) for r in df.itertuples(index=False)}


def is_finished_race(state: ReplayState, dn: int, t: Optional[pd.Timestamp] = None) -> bool:
    dn = int(dn)
    if t is not None and dn in state.finished_at:
        return t >= state.finished_at[dn]
    return dn in state.finished_drivers

def build_gap_closing_windows(
    driver_t: pd.DataFrame,
    pit_windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    *,
    min_gain_s: float = 2.0,
    min_duration_s: int = 3,
    min_gain_rate_s_per_s: float = 0.05,  # ~= 5.0s per 100s (~lap)
    per_step_tolerance_s: float = 0.05,
    max_gap_s: Optional[float] = 60.0,
    ffill_limit_s: Optional[int] = 180,
    finished_at: Optional[Dict[int, pd.Timestamp]] = None,
    allowed_drivers: Optional[Set[int]] = None,
    pit_linger_s: float = 3.0,
) -> Dict[Tuple[int, int], List[dict]]:
    """
    Detect windows where a trailing car reduces the gap to a leading car.

    - Uses gap_to_leader_s from driver_t (per-second).
    - Resets/terminates windows when either car is in pit_windows.
    - Only records runs that last >= min_duration_s, gain >= min_gain_s,
      and average gain rate >= min_gain_rate_s_per_s. This catches both
      lap-scale (e.g., ~2s/lap) and short bursts (e.g., 1s in 10-20s).
    - ffill_limit_s keeps leader gaps alive after the last packet so late closings still register.
    """
    if driver_t is None or driver_t.empty:
        return {}

    g = driver_t.copy()
    if g.empty:
        return {}

    g["driver_number"] = g["driver_number"].astype(int)
    finished_at = finished_at or {}
    pivot = g.pivot(index="t", columns="driver_number", values="gap_to_leader_s").sort_index()
    if ffill_limit_s is not None:
        pivot = pivot.ffill(limit=ffill_limit_s)
    pivot = pivot.dropna(how="all")
    drivers = list(pivot.columns)
    allowed = set(int(d) for d in allowed_drivers) if allowed_drivers else None

    windows: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    state: Dict[Tuple[int, int], dict] = {}

    def flush(key: Tuple[int, int]):
        st = state.get(key)
        if not st or not st.get("active"):
            return
        start_t = st.get("start_t")
        end_t = st.get("last_t")
        start_gap = st.get("start_gap")
        end_gap = st.get("last_gap")
        if start_t is None or end_t is None or start_gap is None or end_gap is None:
            return
        duration = (end_t - start_t).total_seconds()
        total_gain = start_gap - end_gap
        gain_rate = total_gain / duration if duration > 0 else 0
        if duration >= min_duration_s and total_gain >= min_gain_s and gain_rate >= min_gain_rate_s_per_s:
            windows[key].append(
                {
                    "start": start_t,
                    "end": end_t,
                    "start_gap_s": start_gap,
                    "end_gap_s": end_gap,
                    "total_gain_s": total_gain,
                    "avg_gain_rate_s_per_s": gain_rate,
                }
            )
        st["active"] = False
        st["start_t"] = None
        st["start_gap"] = None

    for t in pivot.index:
        # Skip timestep entirely if no usable data
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                dn_a = int(drivers[i])
                dn_b = int(drivers[j])
                if allowed and (dn_a not in allowed or dn_b not in allowed):
                    continue
                gap_a = pivot.at[t, dn_a]
                gap_b = pivot.at[t, dn_b]

                if pd.isna(gap_a) or pd.isna(gap_b) or gap_a == gap_b:
                    key = (min(dn_a, dn_b), max(dn_a, dn_b))
                    flush(key)
                    if key in state:
                        state[key]["last_gap"] = None
                        state[key]["last_t"] = None
                    continue

                trailing, leading, trailing_gap, leading_gap = (
                    (dn_a, dn_b, gap_a, gap_b) if gap_a > gap_b else (dn_b, dn_a, gap_b, gap_a)
                )
                pair_gap = trailing_gap - leading_gap
                key = (trailing, leading)

                if key not in state:
                    state[key] = {
                        "active": False,
                        "start_t": None,
                        "start_gap": None,
                        "last_gap": None,
                        "last_t": None,
                    }
                st = state[key]

                def _in_pit_strong(dn: int) -> bool:
                    for s, e in pit_windows.get(int(dn), []):
                        if s <= t <= e:
                            return True
                        if t > e and (t - e) <= pd.to_timedelta(pit_linger_s, unit="s"):
                            return True
                    return False

                in_pit = _in_pit_strong(trailing) or _in_pit_strong(leading)
                finished_now = (
                    (trailing in finished_at and t >= finished_at[trailing])
                    or (leading in finished_at and t >= finished_at[leading])
                )
                if in_pit or finished_now or pd.isna(pair_gap) or (max_gap_s is not None and pair_gap > max_gap_s):
                    flush(key)
                    st["last_gap"] = None
                    st["last_t"] = None
                    continue

                prev_gap = st.get("last_gap")
                prev_t = st.get("last_t")

                if prev_gap is None or prev_t is None:
                    st["last_gap"] = float(pair_gap)
                    st["last_t"] = t
                    continue

                delta = prev_gap - pair_gap  # >0 means closing

                if st.get("active"):
                    if delta >= -per_step_tolerance_s:
                        st["last_gap"] = float(pair_gap)
                        st["last_t"] = t
                    else:
                        flush(key)
                        st["last_gap"] = float(pair_gap)
                        st["last_t"] = t
                    continue

                if delta > per_step_tolerance_s:
                    st["active"] = True
                    st["start_t"] = prev_t
                    st["start_gap"] = prev_gap
                    st["last_gap"] = float(pair_gap)
                    st["last_t"] = t
                else:
                    st["last_gap"] = float(pair_gap)
                    st["last_t"] = t

    # flush any active windows at the end
    for key in list(state.keys()):
        flush(key)

    # # export to per-second map
    # per_second_map: Dict[pd.Timestamp, List[dict]] = {}
    # for pair, ws in windows.items():
    #     for w in ws:
    #         start = w.get("start")
    #         end = w.get("end")
    #         if pd.isna(start) or pd.isna(end):
    #             continue
    #         for t in pd.date_range(start, end, freq="1s"):
    #             if t not in per_second_map:
    #                 per_second_map[t] = []
    #             per_second_map[t].append(pair)

    # #export to file for viewing
    
    # with open("gap_closing_per_second.json", "w") as f:
    #     json.dump({str(k): v for k, v in per_second_map.items()}, f, indent=2)
    return windows


def build_event_maps(
    rc_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    ov_df: Optional[pd.DataFrame] = None,
):
    rc_map = event_map(rc_df, "race_control")
    pit_map = event_map(pit_df, "pit")
    pos_map = event_map(pos_df, "position")
    ov_map = event_map(ov_df, "overtake") if ov_df is not None else None
    return rc_map, pit_map, pos_map, ov_map

def _extract_pair(e: dict) -> Optional[Tuple[int, int]]:
    a = e.get("driver_number") or e.get("overtaking_driver_number")
    v = e.get("overtaken_driver_number") or e.get("victim")
    if pd.isna(a) or pd.isna(v):
        return None
    return (int(a), int(v))


def _extract_overtake_info(e: dict) -> Optional[dict]:
    """Extract overtake pair with zone information if available."""
    a = e.get("driver_number") or e.get("overtaking_driver_number")
    v = e.get("overtaken_driver_number") or e.get("victim")
    if pd.isna(a) or pd.isna(v):
        return None
    return {
        "pair": (int(a), int(v)),
        "zone": e.get("zone"),
        "zone_type": e.get("zone_type"),
    }


def replay(
    driver_t: pd.DataFrame,
    synthetic_interval_long: pd.DataFrame,
    rc_map,
    pit_map,
    pos_map,
    ov_map=None,
    state: Optional[ReplayState] = None,
    *,
    overtake_storm_count: int = 6,
    return_event_summary: bool = False,
    gap_closing_map: Optional[Dict[pd.Timestamp, List[Tuple[int, int]]]] = None,
    gap_closing_starts: Optional[Dict[pd.Timestamp, List[Tuple[int, int]]]] = None,
    finished_drivers: Optional[Set[int]] = None,
    # speed_lookup: Optional[pd.Series] = None,
) -> Any:
    if driver_t is None or driver_t.empty:
        empty = pd.DataFrame()
        return (empty, {}) if return_event_summary else empty

    state = state or ReplayState()
    if finished_drivers:
        state.finished_drivers = set(finished_drivers)
    finished_at = state.finished_at or {}
    t_index = list(driver_t["t"].dropna().sort_values().unique())
    drivers = [int(d) for d in pd.Series(driver_t["driver_number"]).dropna().unique()]
    gap_lookup = driver_t.set_index(["t", "driver_number"])["gap_to_leader_s"]
    interval_lookup = (
        synthetic_interval_long.set_index(["t", "driver_number"])["interval_synth_s"]
        if synthetic_interval_long is not None and not synthetic_interval_long.empty
        else pd.Series(dtype="float64")
    )
    
    snapshots: List[dict] = []
    event_summary: Dict[pd.Timestamp, dict] = {}

    for t in t_index:
        events: List[dict] = []
        for mapping in (rc_map, pit_map, pos_map, ov_map or {}):
            if mapping:
                events.extend(mapping.get(t, []))

        apply_events(state, t, events)

        pit_events = [e for e in events if e.get("kind") == "pit"]
        overtake_events = [e for e in events if e.get("kind") == "overtake"]

        stormy = len(overtake_events) >= overtake_storm_count
        closing_pairs = gap_closing_map.get(t, []) if gap_closing_map else []
        # drop pairs where either driver is in/just-exited pit (extra guard)
        closing_pairs = [
            pair
            for pair in closing_pairs
            if not (is_in_pit_buffered(state, pair[0], t) or is_in_pit_buffered(state, pair[1], t))
        ]
        race_over = state.track_flag and str(state.track_flag).lower().startswith("chequer")
        if race_over:
            closing_pairs = []
        finished_set = state.finished_drivers
        finished_now = sorted(
            [(dn, state.finished_position.get(dn)) for dn, ft in finished_at.items() if ft == t],
            key=lambda x: (x[1] if x[1] is not None else 999, x[0]),
        )

        # classify overtakes (skip if race is over - cooldown lap overtakes are not relevant)
        real_overtakes: List[dict] = []  # with zone info
        ctx_overtakes: List[dict] = []   # with zone info
        real_pairs: List[Tuple[int, int]] = []  # for backwards compat
        ctx_pairs: List[Tuple[int, int]] = []

        if race_over:
            overtake_events = []

        for e in overtake_events:
            info = _extract_overtake_info(e)
            if info is None:
                continue
            pair = info["pair"]
            a, v = pair
            if stormy or is_bad_context(state, a, t) or is_bad_context(state, v, t):
                ctx_pairs.append(pair)
                ctx_overtakes.append(info)
            else:
                real_pairs.append(pair)
                real_overtakes.append(info)

        # dedupe pairs per second (keep first occurrence with zone info)
        seen_real = set()
        deduped_real = []
        for ov in real_overtakes:
            if ov["pair"] not in seen_real:
                seen_real.add(ov["pair"])
                deduped_real.append(ov)
        real_overtakes = deduped_real
        real_pairs = sorted(set(real_pairs))
        
        seen_ctx = set()
        deduped_ctx = []
        for ov in ctx_overtakes:
            if ov["pair"] not in seen_ctx:
                seen_ctx.add(ov["pair"])
                deduped_ctx.append(ov)
        ctx_overtakes = deduped_ctx
        ctx_pairs = sorted(set(ctx_pairs))

        pit_drivers = sorted({int(e.get("driver_number")) for e in pit_events if pd.notna(e.get("driver_number"))})
        # speed_map = (
        #     {int(dn): speed_lookup.get((t, dn), pd.NA) for dn in drivers}
        #     if speed_lookup is not None
        #     else {}
        # )

        if return_event_summary:
            event_summary[t] = {
                "pit": pit_drivers,
                "real_pairs": real_pairs,
                "real_overtakes": real_overtakes,  # with zone info
                "ctx_pairs": ctx_pairs,
                "ctx_overtakes": ctx_overtakes,    # with zone info
                "storm": stormy,
                "gap_closing": gap_closing_starts.get(t, []) if gap_closing_starts else [],
                "finished": finished_now,
                # "sx_speed_kmh": speed_map,
            }

        # Per-driver snapshot rows (still useful for tower)
        pit_set = set(pit_drivers)
        real_for = {a for a, _ in real_pairs}
        real_against = {v for _, v in real_pairs}
        ctx_for = {a for a, _ in ctx_pairs}
        ctx_against = {v for _, v in ctx_pairs}
        closing_for = {a for a, _ in closing_pairs}
        closing_against = {v for _, v in closing_pairs}

        for dn in drivers:
            gap = gap_lookup.get((t, dn), pd.NA)
            interval_val = interval_lookup.get((t, dn), pd.NA) if not interval_lookup.empty else pd.NA
            # speed_kmh = speed_map.get(dn, pd.NA)

            snapshots.append(
                {
                    "t": t,
                    "driver_number": dn,
                    "gap_to_leader_s": gap,
                    "interval_synth_s": interval_val,
                    "position": state.last_position.get(dn),
                    "track_flag": state.track_flag,
                    "driver_flag": state.driver_flag.get(dn),

                    "in_pit": is_in_pit(state, dn, t),
                    "is_slow": is_slow(state, dn, t),
                    "pit_event": dn in pit_set,

                    "overtake_real_for": dn in real_for,
                    "overtake_real_against": dn in real_against,
                    "overtake_ctx_for": dn in ctx_for,
                    "overtake_ctx_against": dn in ctx_against,
                    "overtake_storm": stormy,
                    "closing_gap_for": dn in closing_for,
                    "closing_gap_against": dn in closing_against,
                    "finished": (dn in finished_at and t >= finished_at[dn]) or dn in finished_set,
                    # "sx_speed_kmh": speed_kmh,
                }
            )

    df = pd.DataFrame(snapshots)
    return (df, event_summary) if return_event_summary else df


async def run_replay(
    session_key: str,
    *,
    cache: bool = True,
    include_overtakes: bool = False,
    include_overtake_location: bool = False,
    state: Optional[ReplayState] = None,
    return_event_summary: bool = False,
):
    state = state or ReplayState()
    async with aiohttp.ClientSession() as session:
        driver_t, synthetic_interval_long = await get_session_intervals(session, session_key, cache=cache)
        rc_df = await get_race_control(session, session_key, cache=cache)
        pit_df = await get_pit(session, session_key, cache=cache)
        pos_df = await get_position(session, session_key, cache=cache)
        ov_df = await get_overtakes(session, session_key, cache=cache) if include_overtakes else None
        laps_df = await get_laps(session, session_key, cache=cache)
        session_result_df = await get_session_result(session, session_key, cache=cache)
        loc_df = await get_location(session, session_key, cache=cache) if include_overtake_location and include_overtakes else None
        # car_df = await get_car_data(session, session_key, cache=cache)

    # Enrich overtakes with location if requested
    if ov_df is not None and loc_df is not None and not ov_df.empty and not loc_df.empty:
        from services.track_zones import enrich_overtakes_with_location
        ov_df = enrich_overtakes_with_location(ov_df, loc_df)

    state.pit_windows = build_pit_windows(pit_df, pre_buffer_s=2.0, post_buffer_s=25.0)

    # Chequered flag time (use to trigger finish events together)
    chequered_t: Optional[pd.Timestamp] = None
    if rc_df is not None and not rc_df.empty:
        rc_track = rc_df[(rc_df.get("scope") == "Track") & (rc_df.get("category") == "Flag")]
        if not rc_track.empty and "flag" in rc_track.columns:
            flags = rc_track.dropna(subset=["flag"])
            flags["flag_lower"] = flags["flag"].astype(str).str.lower()
            cheq = flags[flags["flag_lower"].str.contains("chequer|checkered|chequered", regex=True)]
            if not cheq.empty:
                chequered_t = pd.to_datetime(cheq.sort_values("t")["t"].iloc[0])

    drivers_from_gap: Set[int] = set()
    clean_gap = pd.DataFrame()
    if driver_t is not None and not driver_t.empty:
        clean_gap = driver_t.dropna(subset=["t", "driver_number", "gap_to_leader_s"]).astype({"driver_number": int})
        drivers_from_gap = set(int(d) for d in clean_gap["driver_number"].dropna().unique())

    # Finished timing: prefer position feed (last timestamp seen), fallback to gap feed
    state.finished_at = {}
    pos_clean = None
    best_gap_finish_t: Optional[pd.Timestamp] = None
    if pos_df is not None and not pos_df.empty:
        pos_clean = pos_df.dropna(subset=["t", "driver_number", "position"]).astype({"driver_number": int})
        last_seen_pos = pos_clean.groupby("driver_number")["t"].max()
        state.finished_at = last_seen_pos.to_dict()
    elif driver_t is not None and not driver_t.empty:
        clean = driver_t.dropna(subset=["t", "driver_number", "gap_to_leader_s"]).astype({"driver_number": int})
        if not clean.empty:
            counts = clean.groupby("t")["driver_number"].nunique()
            if not counts.empty:
                best_gap_finish_t = counts.idxmax()
                state.finished_at = {int(r.driver_number): best_gap_finish_t for r in clean[clean["t"] == best_gap_finish_t].itertuples(index=False)}

    state.finished_drivers = build_finished_drivers(session_result_df) or set(state.finished_at.keys())
    state.finished_position = build_finished_positions(session_result_df)

    # Fallback finishing positions: use final position snapshot, then gap ordering
    if not state.finished_position:
        if pos_clean is not None and not pos_clean.empty:
            latest_per_driver = (
                pos_clean.sort_values("t")
                .drop_duplicates(subset=["driver_number"], keep="last")
                .dropna(subset=["position"])
            )
            final_pos = latest_per_driver.sort_values("position")
            state.finished_position = {int(r.driver_number): int(r.position) for r in final_pos.itertuples(index=False)}
        elif not clean_gap.empty:
            if best_gap_finish_t is None:
                counts = clean_gap.groupby("t")["driver_number"].nunique()
                if not counts.empty:
                    best_gap_finish_t = counts.idxmax()
            last_t = best_gap_finish_t if best_gap_finish_t is not None else clean_gap["t"].max()
            derived = (
                clean_gap.loc[clean_gap["t"] == last_t, ["driver_number", "gap_to_leader_s"]]
                .dropna()
                .sort_values("gap_to_leader_s", na_position="last")
            )
            state.finished_position = {int(r.driver_number): i + 1 for i, r in enumerate(derived.itertuples(index=False))}

    # If derived list is shorter than available drivers, rebuild from gap feed
    if not clean_gap.empty and drivers_from_gap and len(state.finished_position) < len(drivers_from_gap):
        target_t = best_gap_finish_t
        if target_t is None:
            counts = clean_gap.groupby("t")["driver_number"].nunique()
            if not counts.empty:
                target_t = counts.idxmax()
        if target_t is None:
            target_t = clean_gap["t"].max()
        derived = (
            clean_gap.loc[clean_gap["t"] == target_t, ["driver_number", "gap_to_leader_s"]]
            .dropna()
            .sort_values("gap_to_leader_s", na_position="last")
        )
        state.finished_position = {int(r.driver_number): i + 1 for i, r in enumerate(derived.itertuples(index=False))}

    # Normalize finish timestamps to the end of the timing stream so finish events fire together
    # and avoid spurious mid-race "finished" when a driver simply drops off the feed.
    # If we have a chequered flag timestamp, align finish detection to it
    if chequered_t is not None and drivers_from_gap:
        state.finished_at = {int(dn): chequered_t for dn in drivers_from_gap}
    elif state.finished_position and not clean_gap.empty:
        common_finish_t = pd.to_datetime(best_gap_finish_t if best_gap_finish_t is not None else clean_gap["t"].max())
        state.finished_at = {int(dn): common_finish_t for dn in state.finished_position.keys()}

    top_finishers = {dn for dn, pos in state.finished_position.items() if pos and pos <= 10}

    best_pos_set: Set[int] = set()
    if pos_df is not None and not pos_df.empty:
        pos_clean = pos_df.dropna(subset=["driver_number", "position", "t"]).astype({"driver_number": int, "position": int})
        best_pos = pos_clean.groupby("driver_number")["position"].min()
        best_pos_set = {int(dn) for dn, p in best_pos.items() if p <= 12}
        leaders = {int(r.driver_number) for r in pos_clean[pos_clean["position"] == 1].itertuples(index=False)}
        best_pos_set |= leaders

    allowed_top = top_finishers or best_pos_set or None

    closing_windows = build_gap_closing_windows(
        driver_t,
        state.pit_windows,
        finished_at=state.finished_at,
        allowed_drivers=allowed_top,
        pit_linger_s=3.0,
    )
    gap_closing_map = _build_window_map_seconds(
        closing_windows,
        per_second=True,
    )
    gap_closing_starts = _build_window_map_seconds(
        closing_windows,
        per_second=False,
        min_gain_s=10.0,  # only show 10+ second gains (major incidents)
        min_gain_rate_s_per_s=0.10,  # at least 0.1s/s gain rate
        include_payload=True,
    )

    state.slow_windows = build_slow_windows_hybrid(
        driver_t,
        synthetic_interval_long,
        pos_df,
        interval_jump_s_per_s=2.0,
        leader_catch_s_per_s=1.5,
        min_run_s=3,
        pad_before_s=1,
        pad_after_s=3,
    )
    rc_map, pit_map, pos_map, ov_map = build_event_maps(rc_df, pit_df, pos_df, ov_df)
    # speed_lookup = build_speed_per_second(car_df, agg="mean")
    
    return replay(
        driver_t,
        synthetic_interval_long,
        rc_map,
        pit_map,
        pos_map,
        ov_map,
        state=state,
        return_event_summary=return_event_summary,
        gap_closing_map=gap_closing_map,
        gap_closing_starts=gap_closing_starts,
        finished_drivers=state.finished_drivers,
        # speed_lookup=speed_lookup,
    )
