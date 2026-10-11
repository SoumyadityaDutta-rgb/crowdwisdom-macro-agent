# CrowdWisdomTrading Macroeconomics Agent
### Quantitative Data Scientist Intern Assessment

An end-to-end Python quantitative machine learning pipeline that ingests macroeconomic calendar releases and economic news sentiment, joins historical trading logs in a relational database, and predicts optimal trading strategy parameter permutations for future periods using rigorous Walk-Forward out-of-sample validation.

---

## 🏛️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │       Macroeconomic Ingestion Layer          │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │  Apify Calendar Scraper   │                                 │    Tavily News Ingester   │
         │  • CPI & Core CPI         │                                 │  • Recent Macro News      │
         │  • FOMC Rate Decisions    │                                 │  • Polarity & Subjectivity│
         │  • Non-Farm Payrolls      │                                 │  • Regime Classification  │
         │  • UTC & EST Alignment    │                                 │  • Lookback: 180 Days     │
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │   SQLite Database Layer      │
                                       │   • macro_events             │
                                       │   • macro_news               │
                                       │   • trading_logs             │
                                       │   • strategy_permutations    │
                                       └──────────────┬───────────────┘
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │ Quantitative Feature Engine  │
                                       │ • Event Proximity (Pre/Post) │
                                       │ • Surprise (Actual - Est)    │
                                       │ • Sentiment Momentum (7d-30d)│
                                       │ • Cyclical Time Encodings    │
                                       └──────────────┬───────────────┘
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │ Walk-Forward ML Validator    │
                                       │ • 4 Out-of-Sample Folds      │
                                       │ • Zero Look-Ahead Bias       │
                                       │ • Gradient Boosted Models    │
                                       │ • Sharpe, Sortino, Drawdown  │
                                       └──────────────┬───────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │ Visual Matrix Outputs     │                                 │ Strategic Deliverables    │
         │ • Sentiment Timeline Plot │                                 │ • 1-Page Macro Outlook    │
         │ • Parameter Matrix / Map  │                                 │ • Model Evaluation Report │
         │ • Optimal Permutation Run │                                 │ • Optimal Future Strategy │
         └───────────────────────────┘                                 └───────────────────────────┘
```

---

## 📦 Project Structure

```
assignment/
├── config/
│   ├── settings.py             # Central configuration, API tokens, paths
│   └── __init__.py
├── data_ingestion/
│   ├── apify_calendar.py       # Apify scraper for 180-day calendar (CPI, FOMC, NFP)
│   ├── tavily_news.py          # Tavily news ingester with financial NLP sentiment
│   ├── trading_logs.py         # Multi-permutation trade logs generator & CSV loader
│   ├── db_manager.py           # Relational SQLite database schema and indices
│   └── __init__.py
├── features/
│   ├── time_features.py        # Cyclical time & event proximity (pre/post 24h)
│   ├── surprise_features.py    # Standardized surprise scores (Actual - Forecast)
│   ├── sentiment_features.py   # Rolling 7d/30d sentiment, momentum & regime
│   ├── feature_pipeline.py     # Leakage-free dataset assembler
│   └── __init__.py
├── models/
│   ├── metrics.py              # Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar
│   ├── walk_forward.py         # Time-series Walk-Forward cross-validation
│   ├── predictor.py            # Gradient Boosted expected PnL & win rate models
│   ├── strategy_selector.py    # Optimal parameter recommendation engine
│   └── __init__.py
├── visualization/
│   ├── sentiment_charts.py     # Macro sentiment timeline chart with event overlays
│   ├── matrix_generator.py     # Parameter performance heatmaps and matrices
│   └── __init__.py
├── reports/
│   ├── macro_summary.py        # 1-page macroeconomic outlook generator
│   ├── report_builder.py       # Evaluation report generator
│   └── MACRO_ECONOMIC_OUTLOOK.md
├── tests/
│   ├── test_ingestion.py       # Data scraper & database tests
│   ├── test_features.py        # Feature engineering & temporal integrity tests
│   └── test_models.py          # Financial metrics & walk-forward tests
├── outputs/                    # Generated charts, heatmaps, and visualizations
├── data/                       # Local SQLite database and cached datasets
├── main.py                     # Master execution pipeline
├── requirements.txt            # Python dependencies
├── .env.example                # Template configuration
├── EVALUATION_REPORT.md        # Comprehensive evaluation report
└── README.md                   # Documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd assignment

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Credentials
Create a `.env` file in the root directory (or copy from `.env.example`):
```env
APIFY_API_TOKEN=your_apify_api_token_here
APIFY_CALENDAR_ACTOR_ID=epctex/trading-economics-scraper
TAVILY_API_KEY=your_tavily_api_key_here
DAYS_LOOKBACK=180
```
> **Offline Fallback Guarantee**: If API tokens are absent or rate-limited, the system automatically falls back to verified macroeconomic schedules and historical news fixtures so that reviewers can execute and evaluate the pipeline seamlessly without interruption.

### 3. Run the Complete Pipeline
```bash
python main.py
```
This executes all 10 stages:
1. Initializes SQLite tables with indexes.
2. Scrapes/ingests 180 days of macroeconomic events via Apify (UTC & EST aligned).
3. Ingests macro news and extracts sentiment via Tavily.
4. Generates/loads multi-permutation strategy trade logs.
5. Builds the quantitative feature matrix.
6. Conducts 4-fold Walk-Forward out-of-sample ML validation.
7. Selects the optimal strategy parameters for the upcoming period.
8. Generates the visual matrix heatmaps and sentiment timeline charts.
9. Compiles the **1-Page Macroeconomic Outlook Summary** (`reports/MACRO_ECONOMIC_OUTLOOK.md`).
10. Compiles the **Quantitative Evaluation Report** (`EVALUATION_REPORT.md`).

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 📊 Deliverables Summary

1. **Macroeconomic Outlook Summary**: See [`reports/MACRO_ECONOMIC_OUTLOOK.md`](reports/MACRO_ECONOMIC_OUTLOOK.md).
2. **Model Evaluation Report**: See [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).
3. **Visual Matrix Outputs**:
   - Sentiment Timeline: [`outputs/macro_sentiment_timeline.png`](outputs/macro_sentiment_timeline.png)
   - Parameter Heatmap: [`outputs/strategy_parameter_matrix.png`](outputs/strategy_parameter_matrix.png)

---

