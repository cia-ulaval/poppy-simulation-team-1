
import asyncio
import base64

from audio.speaker import Speaker
from conversation.prompts import POPPY_SYSTEM_PROMPT
from openai import AsyncOpenAI

from config.settings import REALTIME_MODEL, VOICE


async def main():
    speaker = Speaker()
    audio_chunks = []

    print("Connexion à OpenAI Realtime...")

    async with AsyncOpenAI() as client:
        async with client.realtime.connect(
            model=REALTIME_MODEL
        ) as connection:

            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["audio"],
                    "instructions": POPPY_SYSTEM_PROMPT,
                    "audio": {
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000
                            },
                            "voice": VOICE
                        }
                    }
                }
            )

            print("Session vocale configurée.")

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Présente-toi en français en deux phrases. "
                                "Dis bonjour à ton interlocuteur."
                            )
                        }
                    ]
                }
            )

            await connection.response.create()

            print("Poppy prépare sa réponse...")

            async for event in connection:

                if event.type == "response.output_audio.delta":
                    audio_chunks.append(
                        base64.b64decode(event.delta)
                    )

                elif event.type == "response.output_audio_transcript.delta":
                    print(event.delta, end="", flush=True)

                elif event.type == "response.done":
                    if event.response.status != "completed":
                        raise RuntimeError(
                            f"Réponse incomplète : {event.response.status}"
                        )
                    break

                elif event.type == "error":
                    raise RuntimeError(event.error.message)

    if not audio_chunks:
        raise RuntimeError("Aucun audio reçu d'OpenAI.")

    audio_bytes = b"".join(audio_chunks)

    print("\n\nAudio reçu :", len(audio_bytes), "octets")
    print("Poppy va parler !")

    speaker.play_realtime_audio(audio_bytes)

    print("Lecture terminée !")


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=90))
    except TimeoutError:
        print("\nDélai dépassé.")
    except Exception as error:
        print("\nErreur :", type(error).__name__)
        print("Détail :", error)
