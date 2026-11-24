"""Une conversation avec Poppy : sa voix, son visage, au clavier ou au micro.

Une seule session Realtime reste ouverte pendant toute la conversation : Poppy se
souvient de ce qu'on lui a dit. GPT choisit l'émotion du visage avec l'outil
changer_expression, au début de sa réponse ; la voix est jouée au fur et à mesure,
et la bouche suit son volume (voir audio/lecteur.py et visage.py).

Lancé par chat_clavier.py (clavier seul) et main.py (visage, clavier et micro).
"""

import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from audio.lecteur import LecteurContinu
from config.peripheriques import (
    GPT_SAMPLE_RATE,
    MICROPHONE_DEVICE,
    Reechantillonneur,
    frequence,
)
from conversation.prompts import POPPY_SYSTEM_PROMPT, VISAGE_PROMPT
from visage import EMOTIONS_VISAGE, Visage

# Phrase jouée quand la connexion à OpenAI échoue. Elle s'enregistre une fois, avec
# la clé et le réseau : python enregistrer_hors_ligne.py
SON_HORS_LIGNE = Path(__file__).resolve().parent.parent / "sons" / "hors_ligne.wav"
PHRASE_HORS_LIGNE = "Je n'ai pas de réseau, mais je peux vous faire coucou !"

MICRO_MIN_S = 0.2  # l'API refuse un tampon audio de moins de 100 ms : marge comprise

OUTIL_EXPRESSION = {
    "type": "function",
    "name": "changer_expression",
    "description": "Change l'expression du visage de Poppy sur son écran.",
    "parameters": {
        "type": "object",
        "properties": {"emotion": {"type": "string", "enum": list(EMOTIONS_VISAGE)}},
        "required": ["emotion"],
    },
}


MARGE_ECHO_S = 0.6  # micro ouvert : on n'écoute pas Poppy, ni l'écho juste après lui

# Micro ouvert : l'API repère elle-même le début et la fin de la parole, puis répond.
DETECTION_PAROLE = {
    "type": "server_vad",
    "threshold": 0.5,  # à monter si le bruit de la salle déclenche des réponses
    "prefix_padding_ms": 300,
    "silence_duration_ms": 700,  # silence qui marque la fin de la question
    "create_response": True,
    "interrupt_response": False,  # le micro est coupé pendant que Poppy parle
}


def configuration_session(voice, micro_ouvert=False, interrompre=False):
    """interrompre : on peut couper la parole à Poppy, avec un micro qui ne l'entend pas."""
    entree = {"format": {"type": "audio/pcm", "rate": GPT_SAMPLE_RATE}}
    if micro_ouvert:
        entree["turn_detection"] = {**DETECTION_PAROLE, "interrupt_response": interrompre}
        # Un micro qui n'entend pas Poppy est un casque, ou un micro près de la bouche.
        entree["noise_reduction"] = {"type": "near_field" if interrompre else "far_field"}
        # Pour afficher ce que Poppy a compris ; ne change rien à sa réponse.
        entree["transcription"] = {"model": "gpt-4o-mini-transcribe"}
    else:
        # On appuie pour parler : pas de détection automatique de fin de phrase.
        entree["turn_detection"] = None
    return {
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": POPPY_SYSTEM_PROMPT + VISAGE_PROMPT,
        "audio": {
            "input": entree,
            "output": {
                "format": {"type": "audio/pcm", "rate": GPT_SAMPLE_RATE},
                "voice": voice,
            },
        },
        "tools": [OUTIL_EXPRESSION],
        "tool_choice": "auto",
    }


class Poppy:
    """Une session Realtime ouverte, reliée au haut-parleur et au visage."""

    def __init__(self, connexion, lecteur, visage, interrompre=False):
        self.connexion = connexion
        self.lecteur = lecteur
        self.visage = visage
        self.interrompre = interrompre  # micro ouvert même pendant que Poppy parle
        self._item_audio = None  # message de Poppy en train d'être joué
        self._coupee = False  # réponse interrompue : ignorer la fin de sa voix

    async def poser_texte(self, texte):
        await self.connexion.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": texte}],
        })
        await self._repondre()

    async def poser_au_micro(self):
        """Enregistre jusqu'à ce qu'on appuie sur Entrée, puis fait répondre Poppy."""
        boucle = asyncio.get_running_loop()
        blocs = asyncio.Queue()
        frequence_micro = frequence(MICROPHONE_DEVICE, "input")
        reechantillonneur = Reechantillonneur(frequence_micro, GPT_SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            boucle.call_soon_threadsafe(blocs.put_nowait, indata[:, 0].copy())

        async def envoyer():
            duree = 0.0
            while True:
                bloc = await blocs.get()
                if bloc is None:
                    return duree
                audio = reechantillonneur(bloc)
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                await self.connexion.input_audio_buffer.append(
                    audio=base64.b64encode(pcm16.tobytes()).decode("ascii"))
                duree += len(pcm16) / GPT_SAMPLE_RATE

        await self.connexion.input_audio_buffer.clear()
        envoi = asyncio.create_task(envoyer())
        flux = sd.InputStream(samplerate=frequence_micro, channels=1, dtype="float32",
                              device=MICROPHONE_DEVICE, blocksize=int(frequence_micro * 0.05),
                              callback=callback)
        with flux:
            await asyncio.to_thread(input, "  Je t'écoute... (Entrée pour envoyer) ")
        blocs.put_nowait(None)
        duree = await envoi

        if duree < MICRO_MIN_S:
            await self.connexion.input_audio_buffer.clear()
            print("  Trop court : maintiens la parole un peu plus longtemps.")
            return
        await self.connexion.input_audio_buffer.commit()
        await self._repondre()

    async def _repondre(self):
        """Fait répondre Poppy, et attend que sa réponse soit sortie du haut-parleur."""
        emotion_avant = self.visage.emotion
        self.visage.changer_emotion("thinking")  # jusqu'à ce que GPT choisisse
        self.lecteur.nouvelle_reponse()
        print("Poppy : ", end="", flush=True)
        emotion_choisie = False  # GPT a-t-il choisi un visage pendant ce tour ?

        relance = False
        while True:
            appel, audio_recu = False, False
            if relance:
                # À la relance, l'outil est interdit : sinon GPT peut rappeler
                # changer_expression en boucle au lieu de parler.
                await self.connexion.response.create(response={"tool_choice": "none"})
            else:
                await self.connexion.response.create()
            while True:
                event = await self.connexion.recv()

                if event.type == "response.output_audio.delta":
                    if not audio_recu and not emotion_choisie:
                        # GPT parle sans avoir choisi de visage : on quitte « thinking ».
                        self.visage.changer_emotion(emotion_avant)
                    audio_recu = True
                    self.lecteur.ajouter(base64.b64decode(event.delta))

                elif event.type == "response.output_audio_transcript.delta":
                    print(event.delta, end="", flush=True)

                elif (event.type == "response.output_item.done"
                      and event.item.type == "function_call"):
                    appel = True
                    emotion_choisie = await self._appliquer_outil(event.item) or emotion_choisie

                elif event.type == "response.done":
                    if event.response.status != "completed":
                        print(f"\n  (réponse {event.response.status})", end="")
                    break

                elif event.type == "error":
                    raise RuntimeError(event.error.message)

            # GPT a changé de visage sans parler encore : on le relance pour qu'il
            # parle. Sans cette relance, Poppy change de visage puis se tait.
            if not (appel and not audio_recu) or relance:
                break
            relance = True

        self.lecteur.fin_reponse()
        await self.lecteur.attendre_fin()
        if self.lecteur.premier_son is not None:
            print(f"\n  (premier son après {self.lecteur.premier_son - self.lecteur.debut:.1f} s)")
        else:
            print()

    async def ecouter_en_continu(self):
        """Micro toujours ouvert : on parle quand on veut, Poppy répond à la fin de la phrase.

        Le micro est coupé pendant que Poppy parle, et MARGE_ECHO_S après : sinon il
        s'entendrait dans son propre haut-parleur et se répondrait à lui-même. Avec
        interrompre (un casque, un micro qui ne l'entend pas), le micro reste ouvert :
        parler pendant qu'il parle le fait taire, puis il répond à la nouvelle phrase.
        Taper une question marche aussi.
        """
        boucle = asyncio.get_running_loop()
        frequence_micro = frequence(MICROPHONE_DEVICE, "input")
        reechantillonneur = Reechantillonneur(frequence_micro, GPT_SAMPLE_RATE)
        blocs = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            boucle.call_soon_threadsafe(blocs.put_nowait, indata[:, 0].copy())

        async def envoyer_micro():
            while True:
                bloc = await blocs.get()
                audio = reechantillonneur(bloc)  # toujours, pour garder son état
                if not self.interrompre and self.lecteur.occupe(MARGE_ECHO_S):
                    continue
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                await self.connexion.input_audio_buffer.append(
                    audio=base64.b64encode(pcm16.tobytes()).decode("ascii"))

        async def lire_clavier():
            while True:
                try:
                    texte = (await asyncio.to_thread(input)).strip()
                except EOFError:
                    return
                if texte.lower() in ("quit", "exit", "q"):
                    return
                if texte and not self.lecteur.occupe():
                    await self.poser_texte_sans_attendre(texte)

        await self.connexion.input_audio_buffer.clear()
        flux = sd.InputStream(samplerate=frequence_micro, channels=1, dtype="float32",
                              device=MICROPHONE_DEVICE, blocksize=int(frequence_micro * 0.05),
                              callback=callback)
        print("Micro ouvert : parle quand tu veux. « quit » + Entrée pour sortir.\n")
        with flux:
            taches = [asyncio.create_task(envoyer_micro()),
                      asyncio.create_task(self._suivre_evenements()),
                      asyncio.create_task(lire_clavier())]
            fini, _ = await asyncio.wait(taches, return_when=asyncio.FIRST_COMPLETED)
            for tache in taches:
                tache.cancel()
            for tache in fini:
                tache.result()  # fait remonter une erreur éventuelle

    async def poser_texte_sans_attendre(self, texte):
        await self.connexion.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": texte}],
        })
        await self.connexion.response.create()

    async def _suivre_evenements(self):
        """Réagit aux événements de l'API quand c'est elle qui lance les réponses."""
        emotion_avant = self.visage.emotion
        en_tour = relance = appel = audio_recu = emotion_choisie = False
        fin_parole = None  # instant où l'on a fini de parler : départ de la latence
        # La transcription de la question arrive souvent après le début de la réponse :
        # on n'écrit « Poppy [émotion] : » qu'à son premier mot, pour garder l'ordre.
        ligne_ouverte, emotion_a_afficher = False, ""

        while True:
            event = await self.connexion.recv()

            if event.type == "input_audio_buffer.speech_started":
                if self.interrompre and self.lecteur.occupe():
                    await self._se_taire()
                    ligne_ouverte = False
                print("  (j'écoute...)", flush=True)

            elif event.type == "input_audio_buffer.speech_stopped":
                fin_parole = time.monotonic()
                emotion_avant = self.visage.emotion
                self.visage.changer_emotion("thinking")

            elif event.type == "conversation.item.input_audio_transcription.completed":
                print(("\n" if ligne_ouverte else "") + f"Vous : {event.transcript.strip()}",
                      flush=True)

            elif event.type == "response.created":
                appel = audio_recu = False
                self._coupee = False
                if not en_tour:  # nouvelle question ; une relance continue le même tour
                    self._item_audio = None  # ne pas tronquer la réponse précédente
                    en_tour, relance, emotion_choisie = True, False, False
                    self.lecteur.nouvelle_reponse()
                    if fin_parole is not None:
                        self.lecteur.debut, fin_parole = fin_parole, None
                    ligne_ouverte, emotion_a_afficher = False, ""

            elif event.type == "response.output_audio.delta":
                if self._coupee:  # la fin d'une réponse qu'on vient d'interrompre
                    continue
                if not audio_recu and not emotion_choisie and self.visage.emotion == "thinking":
                    self.visage.changer_emotion(emotion_avant)
                audio_recu = True
                self._item_audio = event.item_id
                self.lecteur.ajouter(base64.b64decode(event.delta))

            elif event.type == "response.output_audio_transcript.delta":
                if self._coupee:
                    continue
                if not ligne_ouverte:
                    print(f"Poppy{emotion_a_afficher} : ", end="", flush=True)
                    ligne_ouverte = True
                print(event.delta, end="", flush=True)

            elif (event.type == "response.output_item.done"
                  and event.item.type == "function_call"):
                appel = True
                choisie = await self._appliquer_outil(event.item, afficher=False)
                if choisie:
                    emotion_choisie, emotion_a_afficher = True, f" [{self.visage.emotion}]"

            elif event.type == "response.done":
                if event.response.status != "completed":
                    print(f"\n  (réponse {event.response.status})", end="")
                coupee = event.response.status == "cancelled"
                if appel and not audio_recu and not relance and not coupee:
                    relance = True  # visage changé sans parler : on le fait parler
                    await self.connexion.response.create(response={"tool_choice": "none"})
                    continue
                en_tour = False
                self.lecteur.fin_reponse()
                asyncio.create_task(self._annoncer_fin())

            elif event.type == "error":
                print(f"\n  Erreur de l'API : {event.error.message}", flush=True)

    async def _se_taire(self):
        """On lui coupe la parole : il se tait, et GPT oublie ce qu'il n'a pas dit.

        L'API arrête d'elle-même la réponse en cours (interrupt_response). Mais elle
        envoie la voix plus vite qu'on ne la joue : sans truncate, GPT croirait qu'on a
        entendu toute sa phrase.
        """
        self._coupee = True
        joue_ms = self.lecteur.interrompre()
        print("\n  (coupé)", flush=True)
        if self._item_audio is not None:
            await self.connexion.conversation.item.truncate(
                item_id=self._item_audio, content_index=0, audio_end_ms=joue_ms)
            self._item_audio = None

    async def _annoncer_fin(self):
        await self.lecteur.attendre_fin()
        if self.lecteur.premier_son is not None:
            print(f"\n  (premier son après {self.lecteur.premier_son - self.lecteur.debut:.1f} s)")
        else:
            print()

    async def _appliquer_outil(self, item, afficher=True):
        if item.name == "changer_expression":
            try:
                emotion = json.loads(item.arguments or "{}").get("emotion")
            except json.JSONDecodeError:
                emotion = None
            choisie = self.visage.changer_emotion(emotion)
            if afficher:  # voir ce que GPT a choisi
                print(f"[{emotion}] ", end="", flush=True)
            resultat = "ok" if choisie else "émotion inconnue"
        else:
            choisie, resultat = False, "outil inconnu"
        await self.connexion.conversation.item.create(item={
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": resultat,
        })
        return choisie


async def hors_ligne(lecteur, visage):
    """Sans connexion, Poppy le dit au lieu de rester muet."""
    visage.changer_emotion("sad")
    print(f"Poppy : {PHRASE_HORS_LIGNE}")
    lecteur.nouvelle_reponse()
    if SON_HORS_LIGNE.exists():
        with wave.open(str(SON_HORS_LIGNE), "rb") as fichier:
            pcm16 = fichier.readframes(fichier.getnframes())
    else:  # phrase pas encore enregistrée : trois bips descendants
        t = np.arange(int(GPT_SAMPLE_RATE * 0.18)) / GPT_SAMPLE_RATE
        bips = [np.sin(2 * np.pi * f * t) * np.hanning(len(t)) for f in (660, 550, 440)]
        silence = np.zeros(int(GPT_SAMPLE_RATE * 0.08))
        signal = np.concatenate([x for bip in bips for x in (bip, silence)])
        pcm16 = (0.3 * signal * 32767).astype(np.int16).tobytes()
    lecteur.ajouter(pcm16)
    lecteur.fin_reponse()
    await lecteur.attendre_fin()


async def discuter(micro=True, micro_ouvert=False, interrompre=False):
    """Conversation. Au clavier : on tape. Au micro : Entrée, ou micro toujours ouvert.

    interrompre (avec micro_ouvert) : parler pendant que Poppy parle le fait taire.
    """
    from openai import AsyncOpenAI

    from config.settings import REALTIME_MODEL, VOICE  # demande la clé

    with Visage("happy") as visage, LecteurContinu(visage) as lecteur:
        print(f"Haut-parleur : {lecteur.nom}")
        if micro or micro_ouvert:
            print(f"Micro : {sd.query_devices(MICROPHONE_DEVICE, 'input')['name']}")
        print("Connexion à OpenAI Realtime...")
        try:
            async with AsyncOpenAI() as client, \
                    client.realtime.connect(model=REALTIME_MODEL) as connexion:
                await connexion.session.update(
                    session=configuration_session(VOICE, micro_ouvert, interrompre))
                poppy = Poppy(connexion, lecteur, visage, interrompre and micro_ouvert)
                if micro_ouvert:
                    await poppy.ecouter_en_continu()
                else:
                    await _boucle(poppy, micro)
        except (OSError, TimeoutError) as erreur:
            print(f"Connexion impossible : {erreur}")
            await hors_ligne(lecteur, visage)
        except Exception as erreur:  # l'API Realtime lève des types variés
            if "invalid_api_key" in str(erreur):  # le serveur ferme la connexion
                print("Clé OpenAI refusée : vérifie OPENAI_API_KEY dans voice_assistant/.env.")
            elif _erreur_reseau(erreur):
                print(f"Connexion impossible : {erreur}")
                await hors_ligne(lecteur, visage)
            else:
                raise


def _erreur_reseau(erreur):
    from openai import APIConnectionError

    if isinstance(erreur, APIConnectionError):
        return True
    module = type(erreur).__module__ or ""
    return module.startswith("websockets") and "InvalidStatus" not in type(erreur).__name__


async def _boucle(poppy, micro):
    if micro:
        consigne = "Vous (Entrée pour parler au micro, ou tapez, « quit » pour sortir) : "
    else:
        consigne = "Vous (« quit » pour sortir) : "
    print("Prêt.\n")
    while True:
        try:
            texte = (await asyncio.to_thread(input, consigne)).strip()
        except EOFError:
            return
        if texte.lower() in ("quit", "exit", "q"):
            return
        try:
            if texte:
                await poppy.poser_texte(texte)
            elif micro:
                await poppy.poser_au_micro()
            else:
                continue
        except RuntimeError as erreur:  # erreur renvoyée par l'API : on continue
            print(f"\n  Erreur de l'API : {erreur}")
            poppy.lecteur.fin_reponse()
