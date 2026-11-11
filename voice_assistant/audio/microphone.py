import numpy as np
import sounddevice as sd


class Microphone:
    def __init__(self, device=1, sample_rate=44100):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = 1

    def record(self, duration=5):
        """Enregistre la voix pendant une durée donnée."""

        print("Poppy écoute... Parle maintenant !")

        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device
        )

        sd.wait()

        print("Enregistrement terminé.")

        return audio.flatten()


if __name__ == "__main__":
    microphone = Microphone()

    audio = microphone.record(duration=5)

    print("Nombre d'échantillons :", len(audio))
    print("Niveau sonore maximal :", np.max(np.abs(audio)))
