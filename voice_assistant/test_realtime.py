
import asyncio

from openai import AsyncOpenAI
from config.settings import REALTIME_MODEL
from conversation.prompts import POPPY_SYSTEM_PROMPT


async def main():
    print("Connexion au serveur Realtime...")

    async with AsyncOpenAI() as client:
        async with client.realtime.connect(
            model=REALTIME_MODEL
        ) as connection:

            print("Connexion établie !")

            # Configurer la personnalité de Poppy
            await connection.session.update(
                session={
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": POPPY_SYSTEM_PROMPT,
                }
            )

            # Envoyer une première question
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Bonjour Poppy ! Présente-toi en français."
                        }
                    ],
                }
            )

            await connection.response.create()

            print("\nPoppy répond :")

            async for event in connection:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)

                elif event.type == "response.done":
                    print("\n")
                    if event.response.status != "completed":
                        print("La réponse ne s'est pas terminée correctement.")
                    break

                elif event.type == "error":
                    print("\nErreur Realtime :", event.error.message)
                    break


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=60))
    except TimeoutError:
        print("\nDélai dépassé : le serveur n'a pas répondu à temps.")
    except Exception as error:
        print("\nErreur :", type(error).__name__)
        print("Détail :", error)
