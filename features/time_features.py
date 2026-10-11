import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

class TimeFeatureEngineer:
    """
    Computes time-based and macroeconomic event proximity features for trade timestamps.
    """

    @staticmethod
    def extract_cyclical_time_features(df: pd.DataFrame, time_col: str = "entry_time") -> pd.DataFrame:
        """Adds cyclical hour and day-of-week sine/cosine encodings."""
        result_df = df.copy()
        dt_series = pd.to_datetime(result_df[time_col])

        result_df["entry_hour"] = dt_series.dt.hour
        result_df["entry_day_of_week"] = dt_series.dt.dayofweek

        # Cyclical transformations
        result_df["sin_hour"] = np.sin(2 * np.pi * result_df["entry_hour"] / 24.0)
        result_df["cos_hour"] = np.cos(2 * np.pi * result_df["entry_hour"] / 24.0)
        result_df["sin_day_of_week"] = np.sin(2 * np.pi * result_df["entry_day_of_week"] / 7.0)
        result_df["cos_day_of_week"] = np.cos(2 * np.pi * result_df["entry_day_of_week"] / 7.0)

        return result_df

    @staticmethod
    def attach_event_proximity(trades_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates hours to next and since last high-impact macro releases
        (CPI, FOMC, Non-Farm Payrolls), and assigns pre/post window indicators.
        """
        result_df = trades_df.copy()
        trades_dt = pd.to_datetime(result_df["entry_time"])

        # Target event categories
        event_types = {
            "fomc": "FOMC",
            "cpi": "CPI",
            "nfp": "Non-Farm Payrolls"
        }

        # Initialize columns
        for key in event_types:
            result_df[f"hours_to_next_{key}"] = 999.0
            result_df[f"hours_since_last_{key}"] = 999.0
        
        result_df["is_event_day"] = 0
        result_df["is_pre_event_window_24h"] = 0
        result_df["is_post_event_window_24h"] = 0

        if events_df.empty:
            return result_df

        events_copy = events_df.copy()
        events_copy["event_dt"] = pd.to_datetime(events_copy["event_timestamp_utc"])
        event_dates_set = set(events_copy["event_dt"].dt.strftime("%Y-%m-%d"))

        # Mark event days
        result_df["is_event_day"] = trades_dt.dt.strftime("%Y-%m-%d").isin(event_dates_set).astype(int)

        for key, name_pattern in event_types.items():
            sub_events = events_copy[events_copy["event_name"].str.contains(name_pattern, case=False, na=False)]
            if sub_events.empty:
                continue

            event_times = sub_events["event_dt"].sort_values().values

            # Vectorized or binary search calculation for each trade
            to_next_list = []
            since_last_list = []

            for t_dt in trades_dt:
                t_val = np.datetime64(t_dt)
                
                # Next event
                future = event_times[event_times >= t_val]
                if len(future) > 0:
                    delta_h = float((future[0] - t_val) / np.timedelta64(1, "h"))
                    to_next_list.append(round(delta_h, 2))
                else:
                    to_next_list.append(999.0)

                # Prior event
                past = event_times[event_times <= t_val]
                if len(past) > 0:
                    delta_h = float((t_val - past[-1]) / np.timedelta64(1, "h"))
                    since_last_list.append(round(delta_h, 2))
                else:
                    since_last_list.append(999.0)

            result_df[f"hours_to_next_{key}"] = to_next_list
            result_df[f"hours_since_last_{key}"] = since_last_list

        # Pre/Post 24h event windows
        min_to_next = result_df[[f"hours_to_next_{k}" for k in event_types]].min(axis=1)
        min_since_last = result_df[[f"hours_since_last_{k}" for k in event_types]].min(axis=1)

        result_df["is_pre_event_window_24h"] = (min_to_next <= 24.0).astype(int)
        result_df["is_post_event_window_24h"] = (min_since_last <= 24.0).astype(int)

        return result_df
