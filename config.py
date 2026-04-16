# config.py
# Central configuration for all StateGen components
# Change settings here — everything else reads from this file

import os

# ─────────────────────────────────────────────
# LLM API
# ─────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "openai_compatible")          # "openai" | "anthropic" | "openai_compatible"
LLM_MODEL         = os.getenv("LLM_MODEL", "deepseek-coder")               # model name for the chosen provider
LLM_BASE_URL      = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")  # DeepSeek endpoint
MAX_NEW_TOKENS    = 2048
TEMPERATURE       = 0.2    # lower = more deterministic
TOP_P             = 0.95

# ─────────────────────────────────────────────
# STATE CONTROLLER
# ─────────────────────────────────────────────
MAX_RETRIES       = 3      # retries per state before giving up

# ─────────────────────────────────────────────
# EXECUTION ENGINE
# ─────────────────────────────────────────────
EXEC_TIMEOUT         = 5.0    # seconds per code run (fresh subprocess)
EXEC_TRACEBACK_LINES = 15     # lines of stderr to include in LLM repair prompts

# ─────────────────────────────────────────────
# MEMORY MANAGER
# ─────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # sentence-transformers
SIM_THRESHOLD     = 0.85                  # cosine similarity cutoff
MEMORY_TOP_K      = 3                     # patterns to inject as few-shot examples
MEMORY_FILE       = "memory/patterns.json"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DS1000_SUBSET     = 200    # problems to evaluate (100 NumPy + 100 Pandas)
BIGCODE_SUBSET    = 50
BIGCODE_USE_INSTRUCT = True   # use shorter instruct_prompt (~663 chars) vs complete_prompt
DATA_DIR          = "data/"
RESULTS_DIR       = "results/"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL         = "INFO"
LOG_FILE          = "logs/stategen.log"

# create directories if they don't exist
for d in ["memory", "data", "results", "logs"]:
    os.makedirs(d, exist_ok=True)
