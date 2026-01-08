"""
Track zone definitions and utilities for determining overtake locations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List

# Silverstone track segments (2020 British GP - session 5768)
# Coordinates are based on OpenF1 location data
SILVERSTONE_ZONES = {
    "Wellington Straight": {"x_range": (-2000, 500), "y_range": (1000, 4500), "type": "straight"},
    "Luffield": {"x_range": (500, 2000), "y_range": (4000, 5000), "type": "corner"},
    "Woodcote": {"x_range": (2000, 4500), "y_range": (4000, 5500), "type": "corner"},
    "Copse": {"x_range": (4500, 6000), "y_range": (5000, 6500), "type": "corner"},
    "Maggots/Becketts": {"x_range": (5500, 7000), "y_range": (4500, 7500), "type": "corner"},
    "Hangar Straight": {"x_range": (3000, 6000), "y_range": (7000, 11000), "type": "straight"},
    "Stowe": {"x_range": (500, 3000), "y_range": (10000, 12000), "type": "corner"},
    "Club": {"x_range": (-1000, 1500), "y_range": (11000, 14000), "type": "corner"},
    "Abbey": {"x_range": (-2500, 0), "y_range": (6000, 11000), "type": "corner"},
    "Farm": {"x_range": (-2500, -1500), "y_range": (2000, 6000), "type": "corner"},
}


def get_track_zone(x: int, y: int, zones: Dict = None) -> Tuple[str, str]:
    """
    Get the track zone name and type (corner/straight) for given coordinates.
    Returns (zone_name, zone_type) or ("Unknown", "unknown").
    """
    zones = zones or SILVERSTONE_ZONES
    for zone_name, bounds in zones.items():
        x_min, x_max = bounds["x_range"]
        y_min, y_max = bounds["y_range"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name, bounds.get("type", "unknown")
    return "Unknown", "unknown"


def get_location_at_time(
    driver_num: int,
    timestamp: pd.Timestamp,
    loc_df: pd.DataFrame,
    tolerance_ms: int = 500,
) -> Optional[Dict]:
    """
    Get the x, y location of a driver at a specific timestamp.
    Returns dict with 'x', 'y' keys or None if no location found within tolerance.
    """
    driver_locs = loc_df[loc_df["driver_number"] == driver_num]
    if driver_locs.empty:
        return None
    
    time_diff = abs((driver_locs["date"] - timestamp).dt.total_seconds() * 1000)
    closest_idx = time_diff.idxmin()
    
    if time_diff[closest_idx] < tolerance_ms:
        row = driver_locs.loc[closest_idx]
        return {"x": int(row["x"]), "y": int(row["y"])}
    return None


def enrich_overtakes_with_location(
    overtakes_df: pd.DataFrame,
    location_df: pd.DataFrame,
    zones: Dict = None,
) -> pd.DataFrame:
    """
    Add location and zone information to overtake events.
    
    Returns DataFrame with added columns:
    - x, y: coordinates
    - zone: track zone name
    - zone_type: 'corner' or 'straight'
    """
    if overtakes_df.empty or location_df.empty:
        return overtakes_df
    
    zones = zones or SILVERSTONE_ZONES
    results = []
    
    for _, ov in overtakes_df.iterrows():
        row = ov.to_dict()
        loc = get_location_at_time(
            ov["overtaking_driver_number"],
            ov["date"],
            location_df,
        )
        if loc:
            row["x"] = loc["x"]
            row["y"] = loc["y"]
            zone_name, zone_type = get_track_zone(loc["x"], loc["y"], zones)
            row["zone"] = zone_name
            row["zone_type"] = zone_type
        else:
            row["x"] = None
            row["y"] = None
            row["zone"] = "Unknown"
            row["zone_type"] = "unknown"
        results.append(row)
    
    return pd.DataFrame(results)


def classify_overtake_location(zone_type: str) -> str:
    """Get a simple description of overtake location."""
    if zone_type == "straight":
        return "on straight"
    elif zone_type == "corner":
        return "into corner"
    return ""
