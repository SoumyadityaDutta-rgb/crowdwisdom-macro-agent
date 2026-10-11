import numpy as np
import pandas as pd
from typing import Optional

class SentimentFeatureEngineer:
    """
    Computes time-decayed rolling macroeconomic news sentiment,
    sentiment momentum, and categorical economic regimes.
    """

    @staticmethod
    def attach_macro_sentiment(trades_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches rolling sentiment signals to trades strictly
        using news published before the trade entry time.
        """
        result_df = trades_df.copy()

        # Defaults
        result_df["sentiment_7d_mean"] = 0.0
        result_df["sentiment_30d_mean"] = 0.0
        result_df["sentiment_momentum"] = 0.0
        result_df["macro_regime"] = "neutral"

        if news_df.empty:
            return result_df

        news_copy = news_df.copy()
        news_copy["news_dt"] = pd.to_datetime(news_copy["published_date"])
        news_sorted = news_copy.sort_values("news_dt")

        trades_dt = pd.to_datetime(result_df["entry_time"])

        sent_7d_list = []
        sent_30d_list = []
        momentum_list = []
        regime_list = []

        for t_dt in trades_dt:
            t_val = np.datetime64(t_dt)
            t_7d_ago = t_val - np.timedelta64(7, "D")
            t_30d_ago = t_val - np.timedelta64(30, "D")

            # 7d window prior news
            news_7d = news_sorted[(news_sorted["news_dt"] <= t_val) & (news_sorted["news_dt"] >= t_7d_ago)]
            score_7d = float(news_7d["sentiment_score"].mean()) if not news_7d.empty else 0.0

            # 30d window prior news
            news_30d = news_sorted[(news_sorted["news_dt"] <= t_val) & (news_sorted["news_dt"] >= t_30d_ago)]
            score_30d = float(news_30d["sentiment_score"].mean()) if not news_30d.empty else 0.0

            # Momentum
            momentum = round(score_7d - score_30d, 3)

            # Macro regime determination
            if score_7d < -0.15 and momentum < 0:
                regime = "hawkish_tightening"
            elif score_7d > 0.15 and momentum >= 0:
                regime = "dovish_easing"
            elif score_7d > 0.20:
                regime = "bullish_expansion"
            elif score_7d < -0.20:
                regime = "recessionary_pressure"
            else:
                regime = "neutral_balanced"

            sent_7d_list.append(round(score_7d, 3))
            sent_30d_list.append(round(score_30d, 3))
            momentum_list.append(momentum)
            regime_list.append(regime)

        result_df["sentiment_7d_mean"] = sent_7d_list
        result_df["sentiment_30d_mean"] = sent_30d_list
        result_df["sentiment_momentum"] = momentum_list
        result_df["macro_regime"] = regime_list

        return result_df
