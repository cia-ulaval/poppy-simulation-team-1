"""Enregistre, une fois pour toutes, la phrase que Poppy dit quand il n'a pas de réseau.

Il faut la clé et Internet pour l'enregistrer ; ensuite, la phrase est jouée depuis
sons/hors_ligne.wav quand la connexion à OpenAI échoue. Sans ce fichier, Poppy fait
trois bips à la place.

Depuis voice_assistant/ : python enregistrer_hors_ligne.py
"""

import asyncio
import base64
import wave

from config.peripheriques import GPT_SAMPLE_RATE
from conversation.assistant import PHRASE_HORS_LIGNE, SON_HORS_LIGNE
from openai import AsyncOpenAI

from config.settings import REALTIME_MODEL, VOICE


async def main():
    morceaux = []
    async with AsyncOpenAI() as client, \
            client.realtime.connect(model=REALTIME_MODEL) as connexion:
        await connexion.session.update(session={
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": "Tu es Poppy, un petit robot humanoïde. Tu lis exactement "
                            "la phrase qu'on te donne, d'un ton déçu mais amusant.",
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": GPT_SAMPLE_RATE},
                                 "voice": VOICE}},
        })
        await connexion.conversation.item.create(item={
            "type": "message", "role": "user",
            "content": [{"type": "input_text",
                         "text": f"Dis exactement, sans rien ajouter : {PHRASE_HORS_LIGNE}"}],
        })
        await connexion.response.create()
        async for event in connexion:
            if event.type == "response.output_audio.delta":
                morceaux.append(base64.b64decode(event.delta))
            elif event.type == "response.done":
                break
            elif event.type == "error":
                raise RuntimeError(event.error.message)

    SON_HORS_LIGNE.parent.mkdir(exist_ok=True)
    with wave.open(str(SON_HORS_LIGNE), "wb") as fichier:
        fichier.setnchannels(1)
        fichier.setsampwidth(2)
        fichier.setframerate(GPT_SAMPLE_RATE)
        fichier.writeframes(b"".join(morceaux))
    print(f"Enregistré : {SON_HORS_LIGNE}")


if __name__ == "__main__":
    asyncio.run(main())
