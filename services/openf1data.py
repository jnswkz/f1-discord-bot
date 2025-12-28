from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    df["speed"] = pd.to_numeric(df.get("speed"), errors="coerce")
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


def is_slow(state: ReplayState, dn: int, t: pd.Timestamp) -> bool:
    return is_in_windows(state.slow_windows, dn, t)


def is_bad_context(state: ReplayState, dn: int, t: pd.Timestamp) -> bool:
    return is_in_pit(state, dn, t) or is_slow(state, dn, t)


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
    # speed_lookup: Optional[pd.Series] = None,
) -> Any:
    if driver_t is None or driver_t.empty:
        empty = pd.DataFrame()
        return (empty, {}) if return_event_summary else empty

    state = state or ReplayState()
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

        # classify overtakes
        real_pairs: List[Tuple[int, int]] = []
        ctx_pairs: List[Tuple[int, int]] = []

        for e in overtake_events:
            pair = _extract_pair(e)
            if pair is None:
                continue
            a, v = pair
            if stormy or is_bad_context(state, a, t) or is_bad_context(state, v, t):
                ctx_pairs.append(pair)
            else:
                real_pairs.append(pair)

        # dedupe pairs per second
        real_pairs = sorted(set(real_pairs))
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
                "ctx_pairs": ctx_pairs,
                "storm": stormy,
                # "sx_speed_kmh": speed_map,
            }

        # Per-driver snapshot rows (still useful for tower)
        pit_set = set(pit_drivers)
        real_for = {a for a, _ in real_pairs}
        real_against = {v for _, v in real_pairs}
        ctx_for = {a for a, _ in ctx_pairs}
        ctx_against = {v for _, v in ctx_pairs}

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
        car_df = await get_car_data(session, session_key, cache=cache)

    state.pit_windows = build_pit_windows(pit_df, pre_buffer_s=2.0, post_buffer_s=25.0)

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
    speed_lookup = build_speed_per_second(car_df, agg="mean")
    
    return replay(
        driver_t,
        synthetic_interval_long,
        rc_map,
        pit_map,
        pos_map,
        ov_map,
        state=state,
        return_event_summary=return_event_summary,
        # speed_lookup=speed_lookup,
    )
