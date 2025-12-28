import asyncio
import os

import pandas as pd

from services.openf1data import run_replay, ReplayState

SESSION_KEY = os.getenv("SESSION_KEY", "5768")

def fmt_drivers(nums):
    return ", ".join(f"#{n}" for n in nums)


def fmt_pairs(pairs):
    # pairs: [(a,v), ...]
    return ", ".join(f"#{a}->{v}" for a, v in pairs)


async def main():
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
