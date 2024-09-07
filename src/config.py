from os import getenv
from dotenv import load_dotenv

load_dotenv()


# --- Prometeus ---
PROMETHEUS_SERVER_PORT: int = int(getenv("PROMETHEUS_SERVER_PORT"))

# --- API ---
API_PORT: int               = int(getenv("API_PORT"))
API_WORKERS: int            = int(getenv("API_WORKERS", 1))
API_RELOAD: bool            = getenv("API_RELOAD", 'True').lower() in ('true', '1', 't')
API_LOG_LEVEL: str          = getenv("API_LOG_LEVEL", "info")

# -- Models ---
SCANN_PATH: str             = getenv("SCANN_PATH")
BRUTE_PATH: str             = getenv("BRUTE_PATH")
RANKING_PATH: str           = getenv("RANKING_PATH")
