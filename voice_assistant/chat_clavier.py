"""Parler à Poppy au clavier : on tape une question, il répond à voix haute.

Une seule session reste ouverte : Poppy se souvient de la conversation. C'est l'outil
pour mettre au point sa personnalité (conversation/prompts.py) sans micro. Si le visage
tourne à côté (python src/ui/screen.py --fenetre, depuis la racine), il suit l'émotion
choisie et la bouche bouge avec la voix.

Depuis voice_assistant/ : python chat_clavier.py   (« quit » pour sortir)
"""

import asyncio

from conversation.assistant import discuter

if __name__ == "__main__":
    try:
        asyncio.run(discuter(micro=False))
    except KeyboardInterrupt:
        pass
