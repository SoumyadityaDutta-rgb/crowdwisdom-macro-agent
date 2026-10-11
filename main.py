import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Ensure assignment root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import LOOKBACK_DAYS
from data_ingestion.db_manager import DatabaseManager
from data_ingestion.apify_calendar import ApifyCalendarScraper
from data_ingestion.tavily_news import TavilyNewsIngester
from data_ingestion.trading_logs import TradingLogIngester
from features.feature_pipeline import FeaturePipeline
from models.predictor import StrategyPredictor
from models.strategy_selector import OptimalStrategySelector
from visualization.sentiment_charts import SentimentVisualizer
from visualization.matrix_generator import VisualMatrixGenerator
from reports.macro_summary import MacroSummaryGenerator
from reports.report_builder import EvaluationReportBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MacroAgentPipeline")

def run_pipeline():
    """
    Executes the end-to-end CrowdWisdomTrading Macroeconomics Agent pipeline.
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING CROWDWISDOMTRADING MACROECONOMICS AGENT PIPELINE")
    logger.info("=" * 80)

    # 1. Database Initialization
    logger.info("\n--- STEP 1: INITIALIZING DATABASE ---")
    db = DatabaseManager()

    # 2. Ingest Macroeconomic Calendar via Apify
    logger.info("\n--- STEP 2: MACROECONOMIC CALENDAR INGESTION (APIFY) ---")
    calendar_scraper = ApifyCalendarScraper()
    events_df = calendar_scraper.fetch_calendar(days_lookback=LOOKBACK_DAYS)
    logger.info(f"Loaded {len(events_df)} macroeconomic event releases (UTC & EST aligned).")

    # 3. Ingest Macroeconomic News & Sentiment via Tavily
    logger.info("\n--- STEP 3: MACRO NEWS & SENTIMENT INGESTION (TAVILY) ---")
    news_ingester = TavilyNewsIngester()
    news_df = news_ingester.fetch_news()
    logger.info(f"Loaded {len(news_df)} macro news articles with sentiment scoring.")

    # 4. Generate Strategy Permutations & Historical Trading Logs
    logger.info("\n--- STEP 4: STRATEGY PERMUTATIONS & TRADING LOG INGESTION ---")
    trading_ingester = TradingLogIngester(db_manager=db)
    perms_df, trades_df = trading_ingester.generate_permutations_and_trades(
        symbols=["SPY", "QQQ"],
        days_lookback=LOOKBACK_DAYS
    )
    logger.info(f"Initialized {len(perms_df)} strategy permutations and {len(trades_df)} trade logs.")

    # 5. Feature Engineering
    logger.info("\n--- STEP 5: QUANTITATIVE FEATURE ENGINEERING ---")
    feature_pipeline = FeaturePipeline(db_manager=db)
    feature_df = feature_pipeline.build_feature_dataset(
        trades_df=trades_df,
        events_df=events_df,
        news_df=news_df
    )
    logger.info(f"Engineered feature matrix: {feature_df.shape[0]} rows x {feature_df.shape[1]} columns.")

    # 6. Machine Learning Model & Walk-Forward Validation
    logger.info("\n--- STEP 6: MACHINE LEARNING & WALK-FORWARD OUT-OF-SAMPLE VALIDATION ---")
    predictor = StrategyPredictor()
    ml_summary = predictor.train_and_validate(feature_df, n_splits=4)
    oos_metrics = ml_summary["overall_oos_metrics"]
    
    logger.info("Out-of-Sample Performance Results:")
    logger.info(f"  • Annualized Sharpe Ratio: {oos_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  • Sortino Ratio:          {oos_metrics['sortino_ratio']:.3f}")
    logger.info(f"  • Maximum Drawdown:        {oos_metrics['max_drawdown']:.2f}%")
    logger.info(f"  • Win Rate:                {oos_metrics['win_rate']*100:.2f}%")
    logger.info(f"  • Profit Factor:           {oos_metrics['profit_factor']:.2f}")

    # 7. Optimal Strategy Parameter Selection for Future Period
    logger.info("\n--- STEP 7: OPTIMAL STRATEGY SELECTION FOR FUTURE PERIOD ---")
    selector = OptimalStrategySelector(predictor=predictor, db_manager=db)
    ranked_strategies = selector.select_optimal_parameters(top_k=5)
    best_strategy = ranked_strategies.iloc[0]
    logger.info(f"Optimal Future Strategy: {best_strategy['strategy_id']}")
    logger.info(f"  • Parameters: Fast={best_strategy['param_fast_window']}, Slow={best_strategy['param_slow_window']}, "
                f"SL={best_strategy['param_stop_loss_pct']*100:.0f}%, TP={best_strategy['param_take_profit_pct']*100:.0f}%")
    logger.info(f"  • Forecasted Return: {best_strategy['predicted_return_pct']*100:.2f}%, Win Prob: {best_strategy['predicted_win_prob']*100:.1f}%")

    # 8. Visual Outputs & Charts
    logger.info("\n--- STEP 8: GENERATING VISUAL OUTPUTS & MATRICES ---")
    sentiment_viz = SentimentVisualizer(db_manager=db)
    timeline_path = sentiment_viz.generate_timeline_chart(news_df=news_df, events_df=events_df)
    logger.info(f"Sentiment timeline generated: {timeline_path}")

    matrix_viz = VisualMatrixGenerator(db_manager=db)
    matrix_path = matrix_viz.generate_parameter_heatmaps(trades_df=trades_df)
    logger.info(f"Parameter performance matrix generated: {matrix_path}")

    # 9. 1-Page Macro Economic Outlook Summary
    logger.info("\n--- STEP 9: GENERATING 1-PAGE MACROECONOMIC OUTLOOK ---")
    outlook_gen = MacroSummaryGenerator(db_manager=db)
    outlook_path = outlook_gen.generate_outlook(events_df=events_df, news_df=news_df)
    logger.info(f"Macroeconomic outlook summary generated: {outlook_path}")

    # 10. Evaluation Report Generation
    logger.info("\n--- STEP 10: GENERATING EVALUATION REPORT ---")
    report_builder = EvaluationReportBuilder()
    report_path = report_builder.build_report(ml_summary=ml_summary, ranked_strategies=ranked_strategies)
    logger.info(f"Evaluation report generated: {report_path}")

    logger.info("=" * 80)
    logger.info("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_pipeline()
