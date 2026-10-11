import numpy as np
import pandas as pd
from typing import List, Dict, Any, Generator, Tuple

class WalkForwardValidator:
    """
    Implements expanding-window or rolling-window Walk-Forward Cross-Validation
    to prevent look-ahead bias and evaluate out-of-sample strategy stability.
    """

    def __init__(self, n_splits: int = 4, min_train_ratio: float = 0.4):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio

    def split(
        self, df: pd.DataFrame, time_col: str = "entry_time"
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Generates expanding-window train and test splits ordered by timestamp.
        """
        sorted_df = df.sort_values(time_col).reset_index(drop=True)
        n_samples = len(sorted_df)

        min_train_size = int(n_samples * self.min_train_ratio)
        remaining_samples = n_samples - min_train_size
        test_size = remaining_samples // self.n_splits

        for fold in range(self.n_splits):
            train_end_idx = min_train_size + fold * test_size
            test_end_idx = train_end_idx + test_size if fold < self.n_splits - 1 else n_samples

            train_df = sorted_df.iloc[:train_end_idx]
            test_df = sorted_df.iloc[train_end_idx:test_end_idx]

            fold_info = {
                "fold": fold + 1,
                "train_start": str(train_df[time_col].iloc[0]),
                "train_end": str(train_df[time_col].iloc[-1]),
                "test_start": str(test_df[time_col].iloc[0]),
                "test_end": str(test_df[time_col].iloc[-1]),
                "train_samples": len(train_df),
                "test_samples": len(test_df)
            }

            yield train_df, test_df, fold_info
