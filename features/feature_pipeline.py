import logging
import pandas as pd
from typing import Optional, Tuple

from data_ingestion.db_manager import DatabaseManager
from features.time_features import TimeFeatureEngineer
from features.surprise_features import SurpriseFeatureEngineer
from features.sentiment_features import SentimentFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Unified pipeline that joins macroeconomic calendar events,
    news sentiment, and trading logs into a clean, leakage-free feature matrix.
    """

    FEATURE_COLUMNS = [
        # Strategy parameters
        "param_fast_window",
        "param_slow_window",
        "param_stop_loss_pct",
        "param_take_profit_pct",
        # Cyclical time
        "entry_hour",
        "entry_day_of_week",
        "sin_hour",
        "cos_hour",
        "sin_day_of_week",
        "cos_day_of_week",
        # Event proximity
        "hours_to_next_fomc",
        "hours_since_last_fomc",
        "hours_to_next_cpi",
        "hours_since_last_cpi",
        "hours_to_next_nfp",
        "hours_since_last_nfp",
        "is_event_day",
        "is_pre_event_window_24h",
        "is_post_event_window_24h",
        # Macro surprises
        "latest_cpi_surprise",
        "latest_fomc_surprise",
        "latest_nfp_surprise",
        "composite_surprise_index",
        "is_inflation_shock",
        # News sentiment & regime
        "sentiment_7d_mean",
        "sentiment_30d_mean",
        "sentiment_momentum"
    ]

    TARGET_COLUMNS = [
        "return_pct",
        "is_win",
        "pnl"
    ]

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def build_feature_dataset(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Orchestrates full feature extraction across trades, events, and news.
        """
        logger.info("Executing unified feature engineering pipeline...")

        if trades_df is None:
            trades_df = self.db.get_trading_logs()
        if events_df is None:
            events_df = self.db.get_macro_events()
        if news_df is None:
            news_df = self.db.get_macro_news()

        if trades_df.empty:
            raise ValueError("Trading logs dataset is empty. Run ingestion first.")

        # 1. Cyclical time features
        df = TimeFeatureEngineer.extract_cyclical_time_features(trades_df, time_col="entry_time")

        # 2. Event proximity features
        df = TimeFeatureEngineer.attach_event_proximity(df, events_df)

        # 3. Macro surprises
        df = SurpriseFeatureEngineer.attach_macro_surprises(df, events_df)

        # 4. News sentiment & regime
        df = SentimentFeatureEngineer.attach_macro_sentiment(df, news_df)

        logger.info(f"Feature dataset constructed successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
