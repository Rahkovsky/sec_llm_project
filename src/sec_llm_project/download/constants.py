from sec_llm_project.utils.env_config import get_sec_user_agent

# SEC API Configuration

SEC_API_BASE = "https://www.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_TXT_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}.txt"

# HTTP Configuration
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 3
HTTP_RETRY_DELAY = 1.0

# Default Settings
DEFAULT_UA = get_sec_user_agent()
DEFAULT_DELAY = 0.25

# Output Directories
OUTPUT_BASE = "data/input/10K"

# Form Types
FORM_TYPES = ["10-K", "10-K/A", "20-F"]

# File Extensions
RAW_EXT = ".raw"
TEXT_EXT = ".txt"
