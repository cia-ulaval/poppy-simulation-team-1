
import numpy as np
import sounddevice as sd
from config.peripheriques import GPT_SAMPLE_RATE, SPEAKER_DEVICE, frequence, reechantillonner


class Speaker:
    def __init__(self, device=SPEAKER_DEVICE, sample_rate=None):
        """Par défaut, le haut-parleur du système, à sa propre fréquence."""
        self.device = device
        self.sample_rate = sample_rate or frequence(device, "output")
        self.channels = 1

    def speak_audio(self, audio):
        """Lit un signal audio float32."""

        audio = np.asarray(audio, dtype=np.float32)

        sd.play(
            audio,
            samplerate=self.sample_rate,
            device=self.device
            )

        sd.wait()

    def play_realtime_audio(self, audio_bytes):
        """
        Lit un bloc PCM16 reçu de l'API Realtime.
        Entrée : mono, 24 kHz, PCM16.
        Sortie : mono, PCM16, à la fréquence du haut-parleur.
        """

        # Conversion des octets en échantillons PCM16
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        if audio.size == 0:
            return

        # Rééchantillonnage 24 kHz vers la fréquence du haut-parleur
        audio_hp = reechantillonner(audio, GPT_SAMPLE_RATE, self.sample_rate)

        # Protection contre les dépassements
        audio_hp = np.clip(
            audio_hp,
            -32768,
            32767
        ).astype(np.int16)

        # Lecture sur le haut-parleur
        sd.play(
            audio_hp,
            samplerate=self.sample_rate,
            device=self.device,
            blocking=True
            )

        sd.wait()


if __name__ == "__main__":
    speaker = Speaker()
    nom = sd.query_devices(speaker.device, "output")["name"]
    print(f"Haut-parleur : {nom} ({speaker.sample_rate} Hz)")

    print("Test du haut-parleur...")

    duration = 2
    frequency = 440

    t = np.linspace(
        0,
        duration,
        int(speaker.sample_rate * duration),
        endpoint=False
    )

    audio = 0.2 * np.sin(2 * np.pi * frequency * t)

    speaker.speak_audio(audio)

    print("Test terminé.")
