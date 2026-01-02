import asyncio
import os
from typing import Optional

import aiohttp
import pandas as pd

from services.openf1data import (
    ReplayState,
    get_car_data,
    get_laps,
    get_location,
    get_pit,
    build_pit_windows,
    build_gap_closing_windows,
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


def fmt_finished(items):
    # items: [(driver, position or None), ...]
    out = []
    for dn, pos in items:
        if pos:
            out.append(f"P{pos} #{dn}")
        else:
            out.append(f"#{dn}")
    return ", ".join(out)


def fmt_closing(items):
    parts = []
    for item in items:
        if isinstance(item, dict):
            pair = item.get("pair")
            if not pair:
                continue
            a, b = pair
            gain = item.get("total_gain_s")
            start = item.get("start_gap_s")
            end = item.get("end_gap_s")
            detail = (
                f"(-{gain:.2f}s; {start:.1f}s->{end:.1f}s)"
                if gain is not None and start is not None and end is not None
                else ""
            )
            parts.append(f"#{a}->{b} {detail}".strip())
        else:
            try:
                a, b = item
                parts.append(f"#{a}->{b}")
            except Exception:
                continue
    return ", ".join(parts)


async def fetch_all_endpoints(session: aiohttp.ClientSession, session_key: str):
    coros = {
        "laps": get_laps(session, session_key, cache=True),
        "stints": get_stints(session, session_key, cache=True),
        "team_radio": get_team_radio(session, session_key, cache=True),
        "weather": get_weather(session, session_key, cache=True),
        "car_data": get_car_data(session, session_key, cache=True),
        "location": get_location(session, session_key, cache=True),
        # Use cached bookends if present; fetch if missing
        "starting_grid": get_starting_grid(session, session_key, cache=True),
        "session_result": get_session_result(session, session_key, cache=True),
    }
    results = await asyncio.gather(*coros.values())
    return {name: df for name, df in zip(coros.keys(), results)}


async def main():
    async with aiohttp.ClientSession() as session:
        all_data = await fetch_all_endpoints(session, SESSION_KEY)
        driver_t, _ = await get_session_intervals(session, SESSION_KEY, cache=True)
        pit_df = await get_pit(session, SESSION_KEY, cache=True)

    print("Fetched OpenF1 datasets:")
    for name, df in all_data.items():
        print(f"- {name}: {len(df)} rows")

    pit_windows = build_pit_windows(pit_df, pre_buffer_s=2.0, post_buffer_s=25.0)
    closing_windows = build_gap_closing_windows(driver_t, pit_windows, min_gain_s=1.0, min_duration_s=3)
    print(f"Computed gap-closing windows for {len(closing_windows)} pairs.")
    # show a small sample
    for pair, wins in list(closing_windows.items())[:5]:
        preview = ", ".join(
            f"{w['start']} -> {w['end']} (-{w['total_gain_s']:.2f}s)" for w in wins[:3]
        ) or "none"
        print(f"  #{pair[0]} vs #{pair[1]}: {len(wins)} windows; sample: {preview}")

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
        gap_closing = info.get("gap_closing", [])
        finished = info.get("finished", [])
        slow = slow_by_t.get(t, [])

        if not pit and not real_pairs and not gap_closing and not finished:
            continue

        parts = []
        # if pit:
        #     parts.append(f"pit: {fmt_drivers(pit)}")

        # if real_pairs:
        #     parts.append(f"overtake(real): {fmt_pairs(real_pairs)}")

        # if ctx_pairs:
        #     label = "overtake(ctx-storm)" if storm else "overtake(ctx)"
        #     parts.append(f"{label}: {fmt_pairs(ctx_pairs)}")

        # if gap_closing:
        #     parts.append(f"closing: {fmt_closing(gap_closing)}")

        if finished:
            parts.append(f"finished: {fmt_finished(finished)}")

        # if slow:
        #     # show speed next to driver
        #     parts.append(f"slow: {fmt_drivers(slow)}")

        if parts:
            print("Events:", " | ".join(parts))  # this will sent to LLM for generating commentary


if __name__ == "__main__":
    asyncio.run(main())
