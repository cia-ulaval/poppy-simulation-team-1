
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


class Speaker:
    def __init__(self, device=2, sample_rate=48000):
        self.device = device
        self.sample_rate = sample_rate
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
        Sortie : mono, 48 kHz, PCM16.
        """

        # Conversion des octets en échantillons PCM16
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        if audio.size == 0:
            return

        # Rééchantillonnage 24 kHz vers 48 kHz
        audio_48k = resample_poly(audio, up=2, down=1)

        # Protection contre les dépassements
        audio_48k = np.clip(
            audio_48k,
            -32768,
            32767
        ).astype(np.int16)

        # Lecture sur le haut-parleur USB
        sd.play(
            audio_48k,
            samplerate=self.sample_rate,
            device=self.device,
            blocking=True
            )

        sd.wait()


if __name__ == "__main__":
    speaker = Speaker()

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
