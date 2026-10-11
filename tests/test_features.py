import pytest
import pandas as pd
import numpy as np
from features.time_features import TimeFeatureEngineer
from features.surprise_features import SurpriseFeatureEngineer
from features.sentiment_features import SentimentFeatureEngineer

def test_cyclical_time_features():
    df = pd.DataFrame({
        "entry_time": ["2024-05-01 10:30:00", "2024-05-01 14:45:00"]
    })
    res = TimeFeatureEngineer.extract_cyclical_time_features(df)
    assert "sin_hour" in res.columns
    assert "cos_hour" in res.columns
    assert "entry_day_of_week" in res.columns
    assert res["entry_hour"].iloc[0] == 10

def test_surprise_zero_lookahead():
    trades_df = pd.DataFrame({
        "entry_time": ["2024-05-01 12:00:00"]
    })
    # Event occurring AFTER trade
    future_events = pd.DataFrame([{
        "event_timestamp_utc": "2024-05-02 12:00:00",
        "event_name": "CPI YoY",
        "surprise": 0.5
    }])
    res = SurpriseFeatureEngineer.attach_macro_surprises(trades_df, future_events)
    # Must not see the future surprise
    assert res["latest_cpi_surprise"].iloc[0] == 0.0

    # Event occurring BEFORE trade
    past_events = pd.DataFrame([{
        "event_timestamp_utc": "2024-04-20 12:00:00",
        "event_name": "CPI YoY",
        "surprise": -0.2
    }])
    res2 = SurpriseFeatureEngineer.attach_macro_surprises(trades_df, past_events)
    assert res2["latest_cpi_surprise"].iloc[0] == -0.2

def test_sentiment_momentum():
    trades_df = pd.DataFrame({
        "entry_time": ["2024-05-10 12:00:00"]
    })
    news_df = pd.DataFrame([
        {"published_date": "2024-05-08 12:00:00", "sentiment_score": 0.4},
        {"published_date": "2024-04-25 12:00:00", "sentiment_score": 0.1}
    ])
    res = SentimentFeatureEngineer.attach_macro_sentiment(trades_df, news_df)
    assert "sentiment_7d_mean" in res.columns
    assert "sentiment_30d_mean" in res.columns
    assert res["sentiment_momentum"].iloc[0] > 0
