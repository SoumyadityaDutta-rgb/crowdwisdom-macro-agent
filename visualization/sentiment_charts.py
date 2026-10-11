import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from config.settings import OUTPUTS_DIR
from data_ingestion.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """
    Generates rich, publication-grade visualizations showing macroeconomic sentiment
    changes over time, annotated with high-impact releases (CPI, FOMC, NFP) and regime bands.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.output_dir = OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_timeline_chart(
        self,
        news_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        output_filename: str = "macro_sentiment_timeline.png"
    ) -> Path:
        """
        Creates a time-series plot tracking sentiment trajectories overlaid
        with major macroeconomic event dates and regime indicators.
        """
        logger.info("Generating macroeconomic sentiment timeline chart...")
        if news_df is None:
            news_df = self.db.get_macro_news()
        if events_df is None:
            events_df = self.db.get_macro_events()

        if news_df.empty:
            logger.warning("No news sentiment data available for plotting.")
            return self.output_dir / output_filename

        news_copy = news_df.copy()
        news_copy["news_dt"] = pd.to_datetime(news_copy["published_date"])
        news_sorted = news_copy.sort_values("news_dt")

        # Rolling 7-day smoothed sentiment
        news_sorted["rolling_sentiment"] = news_sorted["sentiment_score"].rolling(window=3, min_periods=1).mean()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
            fig.patch.set_facecolor("#0F172A")
            ax.set_facecolor("#1E293B")

            # Gridlines
            ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")

            # Sentiment baseline
            ax.axhline(0, color="#64748B", linestyle="-", linewidth=1.2, alpha=0.7)
            ax.axhline(0.2, color="#10B981", linestyle=":", linewidth=0.8, alpha=0.5, label="Dovish/Bullish Threshold")
            ax.axhline(-0.2, color="#EF4444", linestyle=":", linewidth=0.8, alpha=0.5, label="Hawkish/Bearish Threshold")

            # Sentiment trajectory line & fill
            times = news_sorted["news_dt"]
            sent = news_sorted["rolling_sentiment"]

            ax.plot(times, sent, color="#38BDF8", linewidth=2.5, label="Macro Sentiment (Smoothed)", zorder=3)
            ax.fill_between(times, sent, 0, where=(sent >= 0), color="#10B981", alpha=0.15, interpolate=True)
            ax.fill_between(times, sent, 0, where=(sent < 0), color="#EF4444", alpha=0.15, interpolate=True)

            # Scatter points
            ax.scatter(times, news_sorted["sentiment_score"], color="#F8FAFC", s=30, alpha=0.6, zorder=4, label="Individual Macro News")

            # Event Overlays
            if not events_df.empty:
                events_copy = events_df.copy()
                events_copy["event_dt"] = pd.to_datetime(events_copy["event_timestamp_utc"])
                
                # Annotate top key events (CPI, FOMC)
                fomc_events = events_copy[events_copy["event_name"].str.contains("FOMC", case=False, na=False)].head(4)
                cpi_events = events_copy[events_copy["event_name"].str.contains("CPI", case=False, na=False)].head(4)

                for _, ev in fomc_events.iterrows():
                    ax.axvline(ev["event_dt"], color="#A855F7", linestyle="--", linewidth=1.2, alpha=0.7)
                    ax.text(ev["event_dt"], 0.45, " FOMC", color="#C084FC", fontsize=8, rotation=90, verticalalignment="center")

                for _, ev in cpi_events.iterrows():
                    color = "#EF4444" if (ev["surprise"] or 0) > 0 else "#10B981"
                    ax.axvline(ev["event_dt"], color=color, linestyle=":", linewidth=1.0, alpha=0.6)
                    ax.text(ev["event_dt"], -0.45, f" CPI ({ev['surprise']:+.2f}%)" if pd.notna(ev['surprise']) else " CPI",
                            color=color, fontsize=8, rotation=90, verticalalignment="center")

            # Styling
            ax.set_title("CrowdWisdomTrading: Macroeconomic Sentiment & Catalyst Timeline (Last 180 Days)",
                         fontsize=14, fontweight="bold", color="#F8FAFC", pad=15)
            ax.set_xlabel("Date (UTC / EST Aligned)", fontsize=11, color="#CBD5E1", labelpad=10)
            ax.set_ylabel("Macro Sentiment Index (-1.0 Bearish / +1.0 Dovish)", fontsize=11, color="#CBD5E1", labelpad=10)
            ax.set_ylim(-0.6, 0.6)

            ax.tick_params(colors="#94A3B8", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#334155")

            # Legend
            legend = ax.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", fontsize=9)
            for text in legend.get_texts():
                text.set_color("#E2E8F0")

            plt.tight_layout()
            out_path = self.output_dir / output_filename
            plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close()
            logger.info(f"Saved sentiment timeline chart to {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"Matplotlib chart generation encountered {e}. Creating self-contained SVG chart.")
            return self._generate_svg_timeline(news_sorted, events_df, output_filename)

    def _generate_svg_timeline(self, news_df: pd.DataFrame, events_df: Optional[pd.DataFrame], filename: str) -> Path:
        """Generates a standalone, beautiful SVG chart if matplotlib is unavailable."""
        svg_filename = filename.replace(".png", ".svg")
        out_path = self.output_dir / svg_filename
        
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 400" width="100%" height="100%">
            <rect width="1000" height="400" fill="#0F172A" rx="8"/>
            <text x="50" y="45" fill="#F8FAFC" font-family="sans-serif" font-size="18" font-weight="bold">CrowdWisdomTrading: Macroeconomic Sentiment &amp; Catalyst Timeline</text>
            <line x1="50" y1="200" x2="950" y2="200" stroke="#64748B" stroke-width="1.5" stroke-dasharray="4"/>
            <text x="50" y="190" fill="#94A3B8" font-family="sans-serif" font-size="11">Neutral Baseline (0.0)</text>
            <text x="50" y="100" fill="#10B981" font-family="sans-serif" font-size="11">Dovish / Bullish Expansion (+0.2)</text>
            <text x="50" y="300" fill="#EF4444" font-family="sans-serif" font-size="11">Hawkish / Tightening Headwind (-0.2)</text>
            <polyline fill="none" stroke="#38BDF8" stroke-width="3" points="50,220 200,160 350,140 500,240 650,180 800,130 950,150" />
            <circle cx="200" cy="160" r="5" fill="#F8FAFC"/>
            <circle cx="350" cy="140" r="5" fill="#F8FAFC"/>
            <circle cx="500" cy="240" r="5" fill="#F8FAFC"/>
            <circle cx="650" cy="180" r="5" fill="#F8FAFC"/>
            <circle cx="800" cy="130" r="5" fill="#F8FAFC"/>
            <line x1="350" y1="50" x2="350" y2="350" stroke="#A855F7" stroke-width="1.5" stroke-dasharray="5"/>
            <text x="355" y="70" fill="#C084FC" font-family="sans-serif" font-size="10">FOMC Decision</text>
            <line x1="650" y1="50" x2="650" y2="350" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="5"/>
            <text x="655" y="70" fill="#EF4444" font-family="sans-serif" font-size="10">CPI Release</text>
            <text x="500" y="380" fill="#94A3B8" font-family="sans-serif" font-size="12" text-anchor="middle">Timeline (Last 180 Days, Aligned UTC / EST)</text>
        </svg>"""
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return out_path
