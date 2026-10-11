import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from data_ingestion.db_manager import DatabaseManager
from data_ingestion.apify_calendar import ApifyCalendarScraper
from data_ingestion.tavily_news import TavilyNewsIngester
from data_ingestion.trading_logs import TradingLogIngester

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_db.sqlite"
    return DatabaseManager(db_path=db_file)

def test_database_initialization(temp_db):
    """Verify that tables are created properly."""
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "macro_events" in tables
        assert "macro_news" in tables
        assert "trading_logs" in tables
        assert "strategy_permutations" in tables

def test_apify_calendar_generator(temp_db):
    """Verify macro calendar event generation and timestamp formatting."""
    scraper = ApifyCalendarScraper()
    scraper.db = temp_db
    df = scraper.fetch_calendar(days_lookback=30)
    
    assert not df.empty
    assert "event_timestamp_utc" in df.columns
    assert "event_timestamp_est" in df.columns
    assert "event_name" in df.columns
    assert "surprise" in df.columns

    # Verify timestamp formats
    pd.to_datetime(df["event_timestamp_utc"])
    pd.to_datetime(df["event_timestamp_est"])

def test_tavily_sentiment_scoring():
    """Verify financial lexicon sentiment analysis."""
    ingester = TavilyNewsIngester()
    
    bullish_text = "The Federal Reserve indicated cooling inflation and rate cut optimism, supporting economic growth."
    score, label = ingester._analyze_sentiment(bullish_text)
    assert score > 0
    assert label == "dovish_bullish"

    bearish_text = "Sticky inflation and rising recession risks trigger hawkish tightening and widespread layoffs."
    score, label = ingester._analyze_sentiment(bearish_text)
    assert score < 0
    assert label == "hawkish_bearish"

def test_trading_log_permutations(temp_db):
    """Verify strategy permutation and trade log generation."""
    ingester = TradingLogIngester(db_manager=temp_db)
    perms_df, trades_df = ingester.generate_permutations_and_trades(days_lookback=30)

    assert len(perms_df) > 0
    assert len(trades_df) > 0
    assert "strategy_id" in trades_df.columns
    assert "pnl" in trades_df.columns
    assert "is_win" in trades_df.columns
