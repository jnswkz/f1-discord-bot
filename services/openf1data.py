import aiohttp
import asyncio
import dotenv
import os
import pandas as pd

dotenv.load_dotenv()

API_BASE_URL = os.getenv('API_BASE_URL')

async def get_session_intervals(session_id: str) -> list:
    url = f"{API_BASE_URL}/intervals?session_key={session_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
            else:
                print(f"Failed to fetch intervals from {url}, status code: {response.status}")
                return pd.DataFrame()  # Return empty DataFrame on failure
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    # coerce to numeric seconds (handles "+1 LAP" etc -> NaN)
    df["gap_to_leader_s"] = pd.to_numeric(df["gap_to_leader"], errors="coerce")
    df["interval_s"] = pd.to_numeric(df["interval"], errors="coerce")

    # bucket to 1-second grid, keep last update per (bucket, driver)
    df["t"] = df["date"].dt.floor("1s")
    df = (
        df.sort_values("date")
        .groupby(["t", "driver_number"], as_index=False)
        .tail(1)
    )

    # wide "state table" for gap_to_leader in seconds
    gap_wide = (
        df.pivot(index="t", columns="driver_number", values="gap_to_leader_s")
        .sort_index()
        .astype("float64")
        .ffill(limit=10)  # carry forward at most 10 seconds
    )

    # synthetic interval to car ahead (derived from gaps each second)
    def intervals_from_gaps(row: pd.Series) -> pd.Series:
        s = pd.to_numeric(row, errors="coerce").dropna().sort_values()
        return s.diff()  # leader => NaN

    synthetic_interval = gap_wide.apply(intervals_from_gaps, axis=1)
    return synthetic_interval

if __name__ == "__main__":
    async def main():
        test_session_id = "5768"
        intervals = await get_session_intervals(test_session_id)
        # print(f"Intervals for session {test_session_id}: {intervals}")
        # synthetic_interval.to_csv("synthetic_intervals.csv")
    asyncio.run(main())