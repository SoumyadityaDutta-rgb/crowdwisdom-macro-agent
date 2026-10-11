import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# API Keys
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Apify Actor configuration
APIFY_CALENDAR_ACTOR_ID = os.getenv("APIFY_CALENDAR_ACTOR_ID", "epctex/trading-economics-scraper")

# Project directories
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Database
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "database.sqlite")))

# Lookback settings
LOOKBACK_DAYS = int(os.getenv("DAYS_LOOKBACK", 180))

# Macro events of interest
TARGET_MACRO_EVENTS = [
    "CPI",
    "Core CPI",
    "FOMC Interest Rate Decision",
    "Non-Farm Payrolls",
    "Unemployment Rate",
    "GDP Growth Rate",
    "PPI",
    "Retail Sales"
]
