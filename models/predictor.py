import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

from features.feature_pipeline import FeaturePipeline
from models.metrics import PerformanceMetrics
from models.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class StrategyPredictor:
    """
    Predictive machine learning pipeline that models expected trade P&L,
    return percentage, and win rates conditional on macroeconomic regime and strategy permutations.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or FeaturePipeline.FEATURE_COLUMNS
        self.regressor = None
        self.classifier = None
        self.feature_importances_ = {}

    def train_and_validate(
        self, df: pd.DataFrame, n_splits: int = 4
    ) -> Dict[str, Any]:
        """
        Executes Walk-Forward validation across historical folds,
        recording out-of-sample prediction accuracy and strategy portfolio metrics.
        """
        logger.info(f"Starting Walk-Forward ML validation with {n_splits} folds...")
        validator = WalkForwardValidator(n_splits=n_splits)

        fold_results = []
        all_oos_actual = []
        all_oos_pred = []
        all_oos_win_actual = []
        all_oos_win_pred = []

        X_full = df[self.feature_cols].fillna(0)
        y_ret_full = df["return_pct"]
        y_win_full = df["is_win"]

        for train_df, test_df, fold_info in validator.split(df):
            X_train = train_df[self.feature_cols].fillna(0)
            y_train_ret = train_df["return_pct"]
            y_train_win = train_df["is_win"]

            X_test = test_df[self.feature_cols].fillna(0)
            y_test_ret = test_df["return_pct"]
            y_test_win = test_df["is_win"]

            # Train Gradient Boosted Regressor for expected return
            reg = HistGradientBoostingRegressor(max_iter=120, max_depth=6, random_state=42)
            reg.fit(X_train, y_train_ret)

            # Train Gradient Boosted Classifier for win probability
            clf = HistGradientBoostingClassifier(max_iter=120, max_depth=6, random_state=42)
            clf.fit(X_train, y_train_win)

            # Predict Out-of-Sample
            pred_ret = reg.predict(X_test)
            pred_win_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test)

            mse = float(mean_squared_error(y_test_ret, pred_ret))
            r2 = float(r2_score(y_test_ret, pred_ret))
            acc = float(accuracy_score(y_test_win, (pred_win_prob >= 0.5).astype(int)))

            try:
                auc = float(roc_auc_score(y_test_win, pred_win_prob))
            except Exception:
                auc = 0.5

            # Calculate OOS Financial Metrics for test fold trades
            test_metrics = PerformanceMetrics.calculate_metrics(y_test_ret)

            fold_info.update({
                "mse": round(mse, 6),
                "r2": round(r2, 4),
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4),
                "oos_sharpe": test_metrics["sharpe_ratio"],
                "oos_sortino": test_metrics["sortino_ratio"],
                "oos_max_drawdown": test_metrics["max_drawdown"],
                "oos_win_rate": test_metrics["win_rate"]
            })
            fold_results.append(fold_info)

            all_oos_actual.extend(y_test_ret.tolist())
            all_oos_pred.extend(pred_ret.tolist())
            all_oos_win_actual.extend(y_test_win.tolist())
            all_oos_win_pred.extend(pred_win_prob.tolist())

        # Fit final production model on full dataset
        self.regressor = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42)
        self.regressor.fit(X_full, y_ret_full)

        self.classifier = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
        self.classifier.fit(X_full, y_win_full)

        # Compute overall out-of-sample performance metrics
        aggregated_metrics = PerformanceMetrics.calculate_metrics(np.array(all_oos_actual))
        
        # Approximate feature importances using RandomForest fit for interpretability
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(X_full, y_ret_full)
        self.feature_importances_ = dict(zip(self.feature_cols, [round(float(v), 4) for v in rf.feature_importances_]))
        
        # Sort importances descending
        self.feature_importances_ = dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))

        summary = {
            "fold_results": fold_results,
            "overall_oos_metrics": aggregated_metrics,
            "overall_oos_r2": round(float(r2_score(all_oos_actual, all_oos_pred)), 4),
            "overall_oos_accuracy": round(float(accuracy_score(all_oos_win_actual, (np.array(all_oos_win_pred) >= 0.5).astype(int))), 4),
            "feature_importances": self.feature_importances_
        }

        logger.info(f"Walk-Forward validation completed. Aggregate OOS Sharpe: {aggregated_metrics['sharpe_ratio']}, Sortino: {aggregated_metrics['sortino_ratio']}, MDD: {aggregated_metrics['max_drawdown']}%")
        return summary

    def predict_strategy_performance(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predicted return and win probabilities for candidate permutations.
        """
        if self.regressor is None or self.classifier is None:
            raise ValueError("Model is not trained yet. Run train_and_validate() first.")

        X = feature_df[self.feature_cols].fillna(0)
        pred_df = feature_df.copy()
        pred_df["predicted_return_pct"] = self.regressor.predict(X)
        pred_df["predicted_win_prob"] = self.classifier.predict_proba(X)[:, 1]
        return pred_df
