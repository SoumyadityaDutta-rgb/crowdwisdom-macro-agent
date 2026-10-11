import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from config.settings import OUTPUTS_DIR
from data_ingestion.db_manager import DatabaseManager
from models.metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class VisualMatrixGenerator:
    """
    Generates quantitative strategy parameter performance matrices and heatmaps,
    visualizing optimal parameter spaces across varying macroeconomic regimes.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.output_dir = OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_parameter_heatmaps(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        output_filename: str = "strategy_parameter_matrix.png"
    ) -> Path:
        """
        Creates publication-ready 2D matrix heatmaps comparing strategy lookbacks
        and risk parameters (Stop Loss vs Take Profit) on Sharpe Ratio and Win Rate.
        """
        logger.info("Generating strategy parameter performance matrices...")
        if trades_df is None:
            trades_df = self.db.get_trading_logs()

        if trades_df.empty:
            logger.warning("No trading logs available for matrix generation.")
            return self.output_dir / output_filename

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Compute metrics aggregated by parameter pairs
            # 1. Fast Window vs Slow Window -> Average Sharpe Ratio
            fast_slow_groups = trades_df.groupby(["param_fast_window", "param_slow_window"])
            sharpe_records = []
            for (fast, slow), grp in fast_slow_groups:
                m = PerformanceMetrics.calculate_metrics(grp["return_pct"])
                sharpe_records.append({
                    "fast_window": fast,
                    "slow_window": slow,
                    "sharpe": m["sharpe_ratio"],
                    "win_rate": m["win_rate"] * 100,
                    "trade_count": m["trade_count"]
                })
            fast_slow_df = pd.DataFrame(sharpe_records)
            sharpe_pivot = fast_slow_df.pivot(index="fast_window", columns="slow_window", values="sharpe")

            # 2. Stop Loss vs Take Profit -> Win Rate %
            sl_tp_groups = trades_df.groupby(["param_stop_loss_pct", "param_take_profit_pct"])
            risk_records = []
            for (sl, tp), grp in sl_tp_groups:
                m = PerformanceMetrics.calculate_metrics(grp["return_pct"])
                risk_records.append({
                    "stop_loss_pct": f"{int(sl*100)}%",
                    "take_profit_pct": f"{int(tp*100)}%",
                    "win_rate": m["win_rate"] * 100,
                    "sharpe": m["sharpe_ratio"]
                })
            risk_df = pd.DataFrame(risk_records)
            risk_pivot = risk_df.pivot(index="stop_loss_pct", columns="take_profit_pct", values="win_rate")

            # Plot Side-by-Side Matrices
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
            fig.patch.set_facecolor("#0F172A")

            # Heatmap 1: Sharpe Ratio by Lookback
            ax1 = axes[0]
            ax1.set_facecolor("#1E293B")
            sns.heatmap(sharpe_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
                        ax=ax1, linewidths=1.5, linecolor="#0F172A",
                        annot_kws={"size": 11, "weight": "bold", "color": "#0F172A"})
            ax1.set_title("Strategy Matrix 1: Sharpe Ratio by Lookbacks\n(Fast vs Slow Moving Average Windows)",
                          fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)
            ax1.set_xlabel("Slow Lookback Window", fontsize=10, color="#CBD5E1", labelpad=8)
            ax1.set_ylabel("Fast Lookback Window", fontsize=10, color="#CBD5E1", labelpad=8)
            ax1.tick_params(colors="#CBD5E1")

            # Heatmap 2: Win Rate by Risk Targets
            ax2 = axes[1]
            ax2.set_facecolor("#1E293B")
            sns.heatmap(risk_pivot, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                        ax=ax2, linewidths=1.5, linecolor="#0F172A",
                        annot_kws={"size": 11, "weight": "bold", "color": "#0F172A"})
            ax2.set_title("Strategy Matrix 2: Win Rate % by Risk Targets\n(Stop Loss % vs Take Profit %)",
                          fontsize=11, fontweight="bold", color="#F8FAFC", pad=12)
            ax2.set_xlabel("Take Profit Target", fontsize=10, color="#CBD5E1", labelpad=8)
            ax2.set_ylabel("Stop Loss Threshold", fontsize=10, color="#CBD5E1", labelpad=8)
            ax2.tick_params(colors="#CBD5E1")

            plt.tight_layout()
            out_path = self.output_dir / output_filename
            plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close()
            logger.info(f"Saved strategy parameter matrices to {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"Matplotlib heatmap generation encountered {e}. Creating HTML/SVG matrix report.")
            return self._generate_html_matrix_report(trades_df, output_filename)

    def _generate_html_matrix_report(self, trades_df: pd.DataFrame, filename: str) -> Path:
        """Fallback rich HTML table matrix output."""
        html_file = self.output_dir / filename.replace(".png", ".html")
        content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CrowdWisdomTrading: Strategy Parameter Matrix</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0F172A; color: #F8FAFC; padding: 30px; }}
        h1 {{ color: #38BDF8; font-size: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1E293B; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #94A3B8; text-transform: uppercase; font-size: 12px; }}
        tr:hover {{ background: #2D3B4F; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .badge-win {{ background: #065F46; color: #34D399; }}
    </style>
</head>
<body>
    <h1>CrowdWisdomTrading: Strategy Parameter Performance Matrix</h1>
    <p style="color: #94A3B8;">Walk-forward out-of-sample parameter evaluation across macroeconomic regimes.</p>
    <table>
        <thead>
            <tr>
                <th>Strategy ID</th>
                <th>Fast Window</th>
                <th>Slow Window</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Win Rate</th>
                <th>Sharpe Ratio</th>
            </tr>
        </thead>
        <tbody>
"""
        grouped = trades_df.groupby("strategy_id")
        for s_id, grp in grouped:
            m = PerformanceMetrics.calculate_metrics(grp["return_pct"])
            first = grp.iloc[0]
            content += f"""
            <tr>
                <td><strong>{s_id}</strong></td>
                <td>{first['param_fast_window']}</td>
                <td>{first['param_slow_window']}</td>
                <td>{first['param_stop_loss_pct']*100:.0f}%</td>
                <td>{first['param_take_profit_pct']*100:.0f}%</td>
                <td><span class="badge badge-win">{m['win_rate']*100:.1f}%</span></td>
                <td><strong>{m['sharpe_ratio']:.2f}</strong></td>
            </tr>
"""
        content += """
        </tbody>
    </table>
</body>
</html>"""
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(content)
        return html_file
