
import os
from pathlib import Path

from dotenv import load_dotenv

# Chemin du dossier voice_assistant
BASE_DIR = Path(__file__).resolve().parent.parent

# Chargement du fichier .env
load_dotenv(BASE_DIR / ".env")

# Clé API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY introuvable. Vérifie le fichier .env.")

# Configuration audio : micro et haut-parleur par défaut du système, ou ceux du .env
# (POPPY_MICRO, POPPY_HAUT_PARLEUR). Voir config/peripheriques.py.
from config.peripheriques import MICROPHONE_DEVICE, SPEAKER_DEVICE  # noqa: E402, F401

AUDIO_CHANNELS = 1

# Configuration du modèle vocal
REALTIME_MODEL = "gpt-realtime-2.1-mini"
VOICE = "marin"
