import os

from dotenv import load_dotenv
from openai import OpenAI

# Charger la clé depuis le fichier .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Clé API introuvable dans le fichier .env")

# Initialiser le client OpenAI
client = OpenAI(api_key=api_key)

try:
    print("Connexion à OpenAI en cours...")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Bonjour Poppy ! Réponds simplement en français : connexion réussie."
    )

    print("\nConnexion OpenAI réussie !")
    print("Réponse de Poppy :", response.output_text)

except Exception as e:
    print("\nUne erreur est survenue.")
    print("Type :", type(e).__name__)
    print("Détail :", str(e))
