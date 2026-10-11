import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from config.settings import LOOKBACK_DAYS
from data_ingestion.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TradingLogIngester:
    """
    Ingests external trading logs or synthesizes systematic multi-permutation
    trading strategy backtests across the historical macroeconomic period.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def ingest_from_csv(self, file_path: str) -> int:
        """
        Ingests an external trading log CSV into the database.
        Expected columns: trade_id, strategy_id, symbol, entry_time, exit_time,
        entry_price, exit_price, direction, pnl, return_pct, is_win
        """
        logger.info(f"Ingesting external trade logs from {file_path}")
        df = pd.read_csv(file_path)
        
        # Standardize required columns
        required_cols = ["trade_id", "strategy_id", "symbol", "entry_time", "exit_time", "pnl", "return_pct", "is_win"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        count = self.db.insert_trading_logs(df)
        logger.info(f"Successfully ingested {count} external trades.")
        return count

    def generate_permutations_and_trades(
        self,
        symbols: Optional[List[str]] = None,
        days_lookback: int = LOOKBACK_DAYS
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Synthesizes a realistic parameter grid of trading strategies (e.g. Trend Following,
        Mean Reversion, Breakout) and generates backtest trade logs over the lookback period.
        """
        symbols = symbols or ["SPY", "QQQ"]
        logger.info(f"Generating strategy permutations & historical trade logs for {symbols} over {days_lookback} days...")

        np.random.seed(42)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_lookback)

        # 1. Define Systematic Strategy Permutation Grid
        permutations = []
        fast_windows = [5, 10, 20]
        slow_windows = [20, 50, 100]
        stop_loss_pcts = [0.01, 0.02, 0.03]
        take_profit_pcts = [0.02, 0.04, 0.06]

        perm_id_counter = 1
        for fast in fast_windows:
            for slow in slow_windows:
                if fast >= slow:
                    continue
                for sl in stop_loss_pcts:
                    for tp in take_profit_pcts:
                        strat_id = f"SMA_F{fast}_S{slow}_SL{int(sl*100)}_TP{int(tp*100)}"
                        permutations.append({
                            "strategy_id": strat_id,
                            "strategy_type": "SMA_CROSSOVER",
                            "param_fast_window": fast,
                            "param_slow_window": slow,
                            "param_stop_loss_pct": sl,
                            "param_take_profit_pct": tp,
                            "description": f"Trend Crossover: Fast={fast}, Slow={slow}, SL={sl*100}%, TP={tp*100}%"
                        })
                        perm_id_counter += 1

        perms_df = pd.DataFrame(permutations)
        self.db.insert_strategy_permutations(perms_df)
        logger.info(f"Initialized {len(perms_df)} strategy permutations in database.")

        # 2. Simulate Realistic Trades for each Permutation
        trades = []
        base_prices = {"SPY": 540.0, "QQQ": 475.0}

        # Date range generation
        cur_time = start_date
        while cur_time < end_date:
            # Skip weekends
            if cur_time.weekday() < 5:
                # 1 to 3 trading opportunities per day across permutations
                for perm in permutations:
                    # Trading frequency probability depends on fast window
                    prob = 0.25 if perm["param_fast_window"] == 5 else (0.15 if perm["param_fast_window"] == 10 else 0.08)
                    if np.random.rand() < prob:
                        symbol = np.random.choice(symbols)
                        base_p = base_prices[symbol]
                        
                        entry_hour = np.random.randint(9, 15)
                        entry_min = np.random.choice([0, 15, 30, 45])
                        entry_dt = cur_time.replace(hour=entry_hour, minute=entry_min, second=0)

                        # Trade duration: 1 to 48 hours
                        hold_hours = float(np.random.choice([2.5, 6.0, 12.0, 24.0, 48.0]))
                        exit_dt = entry_dt + timedelta(hours=hold_hours)
                        if exit_dt > end_date:
                            continue

                        direction = "LONG" if np.random.rand() > 0.45 else "SHORT"
                        entry_price = round(base_p * (1 + np.random.normal(0, 0.015)), 2)
                        
                        sl = perm["param_stop_loss_pct"]
                        tp = perm["param_take_profit_pct"]

                        # Simulated return influenced by strategy edge & parameters
                        edge = (0.003 if perm["param_slow_window"] == 50 else 0.001)
                        trade_return = np.random.normal(edge, 0.02)
                        
                        # Apply stop loss & take profit caps
                        if trade_return <= -sl:
                            trade_return = -sl
                        elif trade_return >= tp:
                            trade_return = tp

                        exit_price = round(entry_price * (1 + trade_return if direction == "LONG" else 1 - trade_return), 2)
                        pnl = round((exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price), 2)
                        is_win = 1 if pnl > 0 else 0

                        trades.append({
                            "trade_id": f"TRD_{uuid.uuid4().hex[:8].upper()}",
                            "strategy_id": perm["strategy_id"],
                            "symbol": symbol,
                            "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "exit_time": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "direction": direction,
                            "pnl": pnl,
                            "return_pct": round(trade_return, 4),
                            "is_win": is_win,
                            "holding_period_hours": hold_hours,
                            "param_fast_window": perm["param_fast_window"],
                            "param_slow_window": perm["param_slow_window"],
                            "param_stop_loss_pct": perm["param_stop_loss_pct"],
                            "param_take_profit_pct": perm["param_take_profit_pct"]
                        })

            cur_time += timedelta(days=1)

        trades_df = pd.DataFrame(trades).sort_values("entry_time")
        count = self.db.insert_trading_logs(trades_df)
        logger.info(f"Generated and stored {count} historical trades across {len(permutations)} strategy permutations.")
        return perms_df, trades_df
