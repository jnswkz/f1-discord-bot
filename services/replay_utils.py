from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def bucket_to_seconds(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Parse timestamps and add a 1-second floor bucket column named ``t``.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, format="mixed")
    df["t"] = df[date_col].dt.floor("1s")
    return df


def event_map(df: pd.DataFrame, kind: str):
    out: Dict[pd.Timestamp, List[dict]] = defaultdict(list)
    if df is None or df.empty:
        return out

    sorted_df = df.sort_values("t")
    for r in sorted_df.itertuples(index=False):
        event = r._asdict()
        t = event.pop("t")
        event["kind"] = kind
        out[t].append(event)
    return out


def process_intervals(
    df: pd.DataFrame,
    forward_fill_limit: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-form interval tables with per-second buckets and synthetic gaps.
    Returns (driver_t, synthetic_interval_long).
    """
    columns_driver = ["t", "driver_number", "gap_to_leader_s"]
    columns_interval = ["t", "driver_number", "interval_synth_s"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns_driver), pd.DataFrame(columns=columns_interval)

    df = bucket_to_seconds(df, "date")
    df["driver_number"] = pd.to_numeric(df.get("driver_number"), errors="coerce").astype("Int64")
    df["gap_to_leader_s"] = pd.to_numeric(df.get("gap_to_leader"), errors="coerce")
    df["interval_s"] = pd.to_numeric(df.get("interval"), errors="coerce")
    df = df.dropna(subset=["driver_number"])
    df = df.sort_values("date").groupby(["t", "driver_number"], as_index=False).tail(1)

    drivers = sorted(df["driver_number"].unique())
    if not drivers:
        return pd.DataFrame(columns=columns_driver), pd.DataFrame(columns=columns_interval)

    t_index = pd.date_range(df["t"].min(), df["t"].max(), freq="1s", tz="UTC")
    base = pd.MultiIndex.from_product([t_index, drivers], names=["t", "driver_number"]).to_frame(index=False)

    driver_t = base.merge(df[["t", "driver_number", "gap_to_leader_s"]], on=["t", "driver_number"], how="left")
    driver_t["gap_to_leader_s"] = driver_t.groupby("driver_number")["gap_to_leader_s"].ffill(limit=forward_fill_limit)
    driver_t = driver_t.sort_values(["t", "driver_number"])

    gap_wide = driver_t.pivot(index="t", columns="driver_number", values="gap_to_leader_s")
    interval_wide = gap_wide.apply(lambda row: row.sort_values().diff().reindex(row.index), axis=1)

    synthetic_interval_long = interval_wide.reset_index().melt(
        id_vars="t",
        var_name="driver_number",
        value_name="interval_synth_s",
    )
    synthetic_interval_long["driver_number"] = synthetic_interval_long["driver_number"].astype("Int64")

    return driver_t, synthetic_interval_long
