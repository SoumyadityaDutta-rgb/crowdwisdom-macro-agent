import numpy as np
import pandas as pd
from typing import Optional

class SurpriseFeatureEngineer:
    """
    Computes macroeconomic surprise metrics (Actual - Forecast)
    and rolling standardized surprise z-scores aligned with trade timestamps.
    """

    @staticmethod
    def attach_macro_surprises(trades_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches the most recent macro surprises (CPI, FOMC, NFP)
        available strictly *prior* to the trade entry time (zero look-ahead bias).
        """
        result_df = trades_df.copy()
        
        # Initialize default columns
        result_df["latest_cpi_surprise"] = 0.0
        result_df["latest_fomc_surprise"] = 0.0
        result_df["latest_nfp_surprise"] = 0.0
        result_df["composite_surprise_index"] = 0.0
        result_df["is_inflation_shock"] = 0

        if events_df.empty:
            return result_df

        events_copy = events_df.copy()
        events_copy["event_dt"] = pd.to_datetime(events_copy["event_timestamp_utc"])
        events_sorted = events_copy.sort_values("event_dt")

        trades_dt = pd.to_datetime(result_df["entry_time"])

        cpi_surprises = []
        fomc_surprises = []
        nfp_surprises = []
        composite_surprises = []
        inflation_shocks = []

        # Filter event subsets
        cpi_events = events_sorted[events_sorted["event_name"].str.contains("CPI", case=False, na=False)]
        fomc_events = events_sorted[events_sorted["event_name"].str.contains("FOMC", case=False, na=False)]
        nfp_events = events_sorted[events_sorted["event_name"].str.contains("Non-Farm|Payrolls", case=False, na=False)]

        for t_dt in trades_dt:
            t_val = np.datetime64(t_dt)

            # Prior CPI
            cpi_prior = cpi_events[cpi_events["event_dt"] <= t_val]
            cpi_val = float(cpi_prior.iloc[-1]["surprise"]) if not cpi_prior.empty and pd.notna(cpi_prior.iloc[-1]["surprise"]) else 0.0

            # Prior FOMC
            fomc_prior = fomc_events[fomc_events["event_dt"] <= t_val]
            fomc_val = float(fomc_prior.iloc[-1]["surprise"]) if not fomc_prior.empty and pd.notna(fomc_prior.iloc[-1]["surprise"]) else 0.0

            # Prior NFP
            nfp_prior = nfp_events[nfp_events["event_dt"] <= t_val]
            nfp_val = float(nfp_prior.iloc[-1]["surprise"]) if not nfp_prior.empty and pd.notna(nfp_prior.iloc[-1]["surprise"]) else 0.0

            # Standardized composite surprise (CPI weighted negative for equities, NFP positive/negative depending on rate expectations)
            # Higher inflation surprise is hawkish headwind
            composite = round(-1.5 * cpi_val - 2.0 * fomc_val + 0.01 * (nfp_val / 20.0), 3)
            
            # Inflation shock indicator (CPI hotter by >= 0.15%)
            shock = 1 if cpi_val >= 0.15 else (-1 if cpi_val <= -0.15 else 0)

            cpi_surprises.append(cpi_val)
            fomc_surprises.append(fomc_val)
            nfp_surprises.append(nfp_val)
            composite_surprises.append(composite)
            inflation_shocks.append(shock)

        result_df["latest_cpi_surprise"] = cpi_surprises
        result_df["latest_fomc_surprise"] = fomc_surprises
        result_df["latest_nfp_surprise"] = nfp_surprises
        result_df["composite_surprise_index"] = composite_surprises
        result_df["is_inflation_shock"] = inflation_shocks

        return result_df
