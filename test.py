import asyncio
import os

import aiohttp
import pandas as pd

from services.openf1data import (
    ReplayState,
    get_car_data,
    get_laps,
    get_location,
    get_session_intervals,
    get_session_result,
    get_starting_grid,
    get_stints,
    get_team_radio,
    get_weather,
    run_replay,
)

SESSION_KEY = os.getenv("SESSION_KEY", "5768")

def fmt_drivers(nums):
    return ", ".join(f"#{n}" for n in nums)


def fmt_pairs(pairs):
    # pairs: [(a,v), ...]
    return ", ".join(f"#{a}->{v}" for a, v in pairs)


async def fetch_all_endpoints(session: aiohttp.ClientSession, session_key: str):
    coros = {
        "laps": get_laps(session, session_key, cache=True),
        "stints": get_stints(session, session_key, cache=True),
        "team_radio": get_team_radio(session, session_key, cache=True),
        "weather": get_weather(session, session_key, cache=True),
        "car_data": get_car_data(session, session_key, cache=True),
        "location": get_location(session, session_key, cache=True),
        # Always re-fetch bookends to avoid stale/missing cache
        "starting_grid": get_starting_grid(session, session_key, cache=False),
        "session_result": get_session_result(session, session_key, cache=False),
    }
    results = await asyncio.gather(*coros.values())
    return {name: df for name, df in zip(coros.keys(), results)}


def derive_positions_from_intervals(driver_t: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """
    Derive an ordered grid/result using gap_to_leader at a given timestamp.
    """
    if driver_t is None or driver_t.empty:
        return pd.DataFrame()
    slice_df = driver_t.loc[driver_t["t"] == ts, ["driver_number", "gap_to_leader_s"]].copy()
    if slice_df.empty:
        return pd.DataFrame()
    slice_df["gap_to_leader_s"] = pd.to_numeric(slice_df["gap_to_leader_s"], errors="coerce")
    ordered = slice_df.sort_values("gap_to_leader_s", na_position="last").reset_index(drop=True)
    ordered["position"] = ordered.index + 1
    return ordered[["position", "driver_number", "gap_to_leader_s"]]


def print_positions(label: str, df: pd.DataFrame):
    if df is None or df.empty:
        print(f"{label}: no data")
        return
    rows = [f"P{int(r.position)} #{int(r.driver_number)} (gap {r.gap_to_leader_s:.3f}s)"
            if pd.notna(r.gap_to_leader_s) else f"P{int(r.position)} #{int(r.driver_number)}"
            for r in df.itertuples(index=False)]
    print(f"{label}: " + " | ".join(rows))


async def main():
    async with aiohttp.ClientSession() as session:
        all_data = await fetch_all_endpoints(session, SESSION_KEY)
        driver_t, _ = await get_session_intervals(session, SESSION_KEY, cache=True)

    print("Fetched OpenF1 datasets:")
    for name, df in all_data.items():
        print(f"- {name}: {len(df)} rows")

    # Fallback: derive starting grid and final order from intervals if missing
    if all_data["starting_grid"].empty or all_data["session_result"].empty:
        if driver_t is None or driver_t.empty:
            print("No interval data to derive grid/result.")
        else:
            first_t = driver_t["t"].min()
            last_t = driver_t["t"].max()
            derived_grid = derive_positions_from_intervals(driver_t, first_t)
            derived_result = derive_positions_from_intervals(driver_t, last_t)
            if all_data["starting_grid"].empty:
                print_positions("Derived starting grid (from first interval)", derived_grid)
            if all_data["session_result"].empty:
                print_positions("Derived session result (from last interval)", derived_result)

    state = ReplayState()
    snapshot_df, events_by_t = await run_replay(
        SESSION_KEY,
        cache=True,
        include_overtakes=True,
        state=state,
        return_event_summary=True,
    )

    slow_by_t = (
        snapshot_df.loc[snapshot_df["is_slow"] == True, ["t", "driver_number"]]
        .dropna()
        .groupby("t")["driver_number"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
        .to_dict()
    )
    
    # Print events per second as PAIRS
    for t in sorted(events_by_t.keys()):
        info = events_by_t[t]

        pit = info["pit"]
        real_pairs = info["real_pairs"]
        ctx_pairs = info["ctx_pairs"]
        storm = info["storm"]
        slow = slow_by_t.get(t, [])

        if not pit and not real_pairs and not ctx_pairs:
            continue

        parts = []
        if pit:
            parts.append(f"pit: {fmt_drivers(pit)}")

        if real_pairs:
            parts.append(f"overtake(real): {fmt_pairs(real_pairs)}")

        if ctx_pairs:
            label = "overtake(ctx-storm)" if storm else "overtake(ctx)"
            parts.append(f"{label}: {fmt_pairs(ctx_pairs)}")


        if slow:
            # show speed next to driver
            parts.append(f"slow: {fmt_drivers(slow)}")
        print("Events:", " | ".join(parts))


if __name__ == "__main__":
    asyncio.run(main())
