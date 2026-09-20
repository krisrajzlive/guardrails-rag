import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"

SOURCE_PDF = DATA_DIR / "vendor_nda.pdf"

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# "openai_moderation" (hosted, no local inference) or "llama_guard" (local transformers)
GUARD_ENGINE = os.getenv("GUARD_ENGINE", "openai_moderation")
GUARD_MODEL_ID = os.getenv("GUARD_MODEL_ID", "meta-llama/Llama-Guard-3-1B")
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "4"))

# Dedicated prompt-injection classifier, hosted via HF Inference API (no local
# download). Neither guard engine's taxonomy covers injection at all -- see
# README "Verified end-to-end" -- so this always runs as a third input-gate
# check regardless of GUARD_ENGINE.
INJECTION_MODEL = os.getenv("INJECTION_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")
INJECTION_THRESHOLD = float(os.getenv("INJECTION_THRESHOLD", "0.75"))
