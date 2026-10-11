import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from data_ingestion.db_manager import DatabaseManager
from models.predictor import StrategyPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class OptimalStrategySelector:
    """
    Selects and ranks optimal strategy parameter permutations for an upcoming future period
    conditioned on predicted or prevailing macroeconomic regimes.
    """

    def __init__(self, predictor: StrategyPredictor, db_manager: Optional[DatabaseManager] = None):
        self.predictor = predictor
        self.db = db_manager or DatabaseManager()

    def select_optimal_parameters(
        self,
        current_macro_state: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Scores all candidate strategy permutations against the forward-looking macro regime
        and identifies the optimal parameter permutation.
        """
        logger.info("Evaluating strategy permutations for future period...")
        
        # Load all strategy permutations
        with self.db.get_connection() as conn:
            perms_df = pd.read_sql_query("SELECT * FROM strategy_permutations", conn)

        if perms_df.empty:
            raise ValueError("No strategy permutations found in database.")

        # Default macro state if not provided (pulled from most recent database entries)
        if current_macro_state is None:
            latest_events = self.db.get_macro_events()
            latest_news = self.db.get_macro_news()

            latest_cpi = latest_events[latest_events["event_name"].str.contains("CPI", na=False)]
            cpi_surprise = float(latest_cpi.iloc[-1]["surprise"]) if not latest_cpi.empty and pd.notna(latest_cpi.iloc[-1]["surprise"]) else 0.0

            latest_sent = float(latest_news["sentiment_score"].mean()) if not latest_news.empty else 0.15

            current_macro_state = {
                "entry_hour": 10,
                "entry_day_of_week": 2,
                "sin_hour": np.sin(2 * np.pi * 10 / 24.0),
                "cos_hour": np.cos(2 * np.pi * 10 / 24.0),
                "sin_day_of_week": np.sin(2 * np.pi * 2 / 7.0),
                "cos_day_of_week": np.cos(2 * np.pi * 2 / 7.0),
                "hours_to_next_fomc": 120.0,
                "hours_since_last_fomc": 240.0,
                "hours_to_next_cpi": 72.0,
                "hours_since_last_cpi": 360.0,
                "hours_to_next_nfp": 96.0,
                "hours_since_last_nfp": 400.0,
                "is_event_day": 0,
                "is_pre_event_window_24h": 0,
                "is_post_event_window_24h": 0,
                "latest_cpi_surprise": cpi_surprise,
                "latest_fomc_surprise": 0.0,
                "latest_nfp_surprise": 25.0,
                "composite_surprise_index": round(-1.5 * cpi_surprise + 0.012, 3),
                "is_inflation_shock": 1 if cpi_surprise > 0.15 else 0,
                "sentiment_7d_mean": round(latest_sent, 3),
                "sentiment_30d_mean": round(latest_sent * 0.8, 3),
                "sentiment_momentum": round(latest_sent * 0.2, 3)
            }

        # Build candidate evaluation dataframe
        candidate_rows = []
        for _, row in perms_df.iterrows():
            cand = dict(row)
            cand.update(current_macro_state)
            candidate_rows.append(cand)

        candidate_df = pd.DataFrame(candidate_rows)

        # Predict expected return and win probability
        scored_df = self.predictor.predict_strategy_performance(candidate_df)

        # Composite Quant Score = Expected Return * (Win Prob / (1 - Win Prob + 1e-4))
        scored_df["expected_quant_score"] = (
            scored_df["predicted_return_pct"] * 100 * (scored_df["predicted_win_prob"] * 1.5)
        )

        ranked_df = scored_df.sort_values("expected_quant_score", ascending=False).reset_index(drop=True)
        ranked_df["rank"] = ranked_df.index + 1

        top_perms = ranked_df.head(top_k)
        best = top_perms.iloc[0]

        logger.info(
            f"Optimal Permutation Selected: '{best['strategy_id']}' "
            f"(Fast={best['param_fast_window']}, Slow={best['param_slow_window']}, "
            f"SL={best['param_stop_loss_pct']*100}%, TP={best['param_take_profit_pct']*100}%) | "
            f"Predicted Return: {best['predicted_return_pct']*100:.2f}%, Win Prob: {best['predicted_win_prob']*100:.1f}%"
        )
        return ranked_df
