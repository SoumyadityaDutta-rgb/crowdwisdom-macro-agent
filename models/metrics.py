import numpy as np
import pandas as pd
from typing import Dict, Any, Union

class PerformanceMetrics:
    """
    Computes rigorous quantitative metrics for financial trading strategies:
    Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, and Profit Factor.
    """

    @staticmethod
    def calculate_metrics(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.04) -> Dict[str, float]:
        """
        Calculates full quantitative performance metrics from a series of trade returns.
        """
        if isinstance(returns, pd.Series):
            ret_arr = returns.dropna().values
        else:
            ret_arr = np.array(returns)
            ret_arr = ret_arr[~np.isnan(ret_arr)]

        if len(ret_arr) == 0:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0
            }

        n_trades = len(ret_arr)
        wins = ret_arr[ret_arr > 0]
        losses = ret_arr[ret_arr < 0]

        win_rate = round(float(len(wins) / n_trades), 4)

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Cumulative equity curve
        equity_curve = np.cumprod(1 + ret_arr)
        total_return_pct = round(float((equity_curve[-1] - 1.0) * 100), 2)

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_drawdown = round(float(np.abs(np.min(drawdowns)) * 100), 2) if len(drawdowns) > 0 else 0.0

        # Annualization factor (assuming ~252 trading days or ~252 trades/year base)
        annualization = np.sqrt(252)

        # Sharpe Ratio
        mean_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr, ddof=1)) if n_trades > 1 else 1e-6
        rf_per_trade = risk_free_rate / 252.0

        if std_ret > 1e-8:
            sharpe_ratio = round(float(((mean_ret - rf_per_trade) / std_ret) * annualization), 3)
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (downside semi-deviation relative to risk-free rate)
        downside_diff = np.minimum(0.0, ret_arr - rf_per_trade)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_dev > 1e-8:
            sortino_ratio = round(float(((mean_ret - rf_per_trade) / downside_dev) * annualization), 3)
        else:
            sortino_ratio = sharpe_ratio

        # Calmar Ratio
        calmar_ratio = round(total_return_pct / max_drawdown, 2) if max_drawdown > 0 else 0.0

        return {
            "trade_count": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio
        }
