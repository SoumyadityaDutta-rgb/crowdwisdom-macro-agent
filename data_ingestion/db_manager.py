import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from config.settings import DATABASE_PATH

class DatabaseManager:
    """Manages the SQLite database operations for macroeconomic events, news, and trade logs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables for macro events, news articles, and trading logs."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Macroeconomic Events table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_timestamp_utc TEXT NOT NULL,
                event_timestamp_est TEXT NOT NULL,
                event_name TEXT NOT NULL,
                country TEXT DEFAULT 'US',
                actual_value REAL,
                forecast_value REAL,
                previous_value REAL,
                unit TEXT,
                surprise REAL,
                source TEXT DEFAULT 'apify',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Macro News and Sentiment table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                published_date TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                source TEXT DEFAULT 'tavily',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Strategy Permutations metadata
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_permutations (
                strategy_id TEXT PRIMARY KEY,
                strategy_type TEXT NOT NULL,
                param_fast_window INTEGER,
                param_slow_window INTEGER,
                param_stop_loss_pct REAL,
                param_take_profit_pct REAL,
                description TEXT
            );
            """)

            # Historical Trading Logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_logs (
                trade_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                direction TEXT NOT NULL,
                pnl REAL NOT NULL,
                return_pct REAL NOT NULL,
                is_win INTEGER NOT NULL,
                holding_period_hours REAL,
                param_fast_window INTEGER,
                param_slow_window INTEGER,
                param_stop_loss_pct REAL,
                param_take_profit_pct REAL,
                FOREIGN KEY (strategy_id) REFERENCES strategy_permutations(strategy_id)
            );
            """)

            # Indexes for high performance queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_events_time ON macro_events(event_timestamp_utc);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_events_name ON macro_events(event_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_news_date ON macro_news(published_date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trading_logs_entry ON trading_logs(entry_time);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trading_logs_strat ON trading_logs(strategy_id);")

            conn.commit()

    def insert_macro_events(self, events_df: pd.DataFrame) -> int:
        """Inserts macro events dataframe into SQLite."""
        if events_df.empty:
            return 0
        with self.get_connection() as conn:
            events_df.to_sql("macro_events", conn, if_exists="append", index=False)
            return len(events_df)

    def insert_macro_news(self, news_df: pd.DataFrame) -> int:
        """Inserts macro news dataframe into SQLite."""
        if news_df.empty:
            return 0
        with self.get_connection() as conn:
            news_df.to_sql("macro_news", conn, if_exists="append", index=False)
            return len(news_df)

    def insert_strategy_permutations(self, perms_df: pd.DataFrame) -> int:
        """Inserts strategy permutation metadata."""
        if perms_df.empty:
            return 0
        with self.get_connection() as conn:
            perms_df.to_sql("strategy_permutations", conn, if_exists="replace", index=False)
            return len(perms_df)

    def insert_trading_logs(self, trades_df: pd.DataFrame) -> int:
        """Inserts historical trading logs."""
        if trades_df.empty:
            return 0
        with self.get_connection() as conn:
            trades_df.to_sql("trading_logs", conn, if_exists="append", index=False)
            return len(trades_df)

    def get_macro_events(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetches macro events within an optional date range."""
        query = "SELECT * FROM macro_events"
        params = []
        conditions = []
        if start_date:
            conditions.append("event_timestamp_utc >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("event_timestamp_utc <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY event_timestamp_utc ASC"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_macro_news(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetches macro news within an optional date range."""
        query = "SELECT * FROM macro_news"
        params = []
        conditions = []
        if start_date:
            conditions.append("published_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("published_date <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY published_date ASC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_trading_logs(self, strategy_id: Optional[str] = None) -> pd.DataFrame:
        """Fetches trading logs, optionally filtered by strategy_id."""
        query = "SELECT * FROM trading_logs"
        params = []
        if strategy_id:
            query += " WHERE strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY entry_time ASC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
