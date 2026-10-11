import pytest
import numpy as np
import pandas as pd
from models.metrics import PerformanceMetrics
from models.walk_forward import WalkForwardValidator

def test_performance_metrics_calculation():
    # 5 wins of +2%, 2 losses of -1%
    returns = np.array([0.02, 0.02, 0.02, -0.01, 0.02, -0.01, 0.02])
    metrics = PerformanceMetrics.calculate_metrics(returns)

    assert metrics["trade_count"] == 7
    assert metrics["win_rate"] == round(5 / 7, 4)
    assert metrics["profit_factor"] == 5.0
    assert metrics["sharpe_ratio"] > 0
    assert metrics["sortino_ratio"] > 0
    assert metrics["max_drawdown"] >= 0

def test_walk_forward_splitter():
    df = pd.DataFrame({
        "entry_time": pd.date_range("2024-01-01", periods=100, freq="D"),
        "return_pct": np.random.normal(0.001, 0.01, 100),
        "is_win": np.random.choice([0, 1], 100)
    })

    validator = WalkForwardValidator(n_splits=3, min_train_ratio=0.5)
    splits = list(validator.split(df))
    assert len(splits) == 3

    for train_df, test_df, fold_info in splits:
        assert len(train_df) > 0
        assert len(test_df) > 0
        # Check temporal order: max train date < min test date
        assert train_df["entry_time"].max() <= test_df["entry_time"].min()
