"""Montée progressive de la difficulté (paliers 2 et 3).

La spec demande de monter le relief et l'intensité des poussées
progressivement : « relief faible au début, plus marqué ensuite ». La
difficulté suit l'avancement de l'entraînement, en pas absolus : un
entraînement repris après une interruption retrouve la difficulté où il
l'avait laissée.
"""

from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class CurriculumConfig:
    """Section ``curriculum:`` du YAML.

    Attributes:
        terrain_difficulty: Difficulté du terrain au début et à la fin de la
            montée, ou ``None`` pour ne pas y toucher.
        push_scale: Multiplicateur des poussées au début et à la fin.
        ramp_fraction: Part de l'entraînement consacrée à la montée ; la
            difficulté reste ensuite au maximum.
    """

    terrain_difficulty: tuple[float, float] | None = None
    push_scale: tuple[float, float] | None = None
    ramp_fraction: float = 0.7

    @classmethod
    def from_yaml(cls, section: dict | None) -> CurriculumConfig | None:
        """Lit la section, ou rend ``None`` si elle est absente."""
        if not section:
            return None
        known = {"terrain_difficulty", "push_scale", "ramp_fraction"}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError(f"Clés inconnues dans curriculum: {', '.join(unknown)}")

        def pair(key):
            value = section.get(key)
            return None if value is None else (float(value[0]), float(value[1]))

        return cls(
            terrain_difficulty=pair("terrain_difficulty"),
            push_scale=pair("push_scale"),
            ramp_fraction=float(section.get("ramp_fraction", 0.7)),
        )

    def at(self, progress: float) -> dict[str, float]:
        """Réglages à ``progress`` (0 au début de l'entraînement, 1 à la fin)."""
        ramp = min(1.0, max(0.0, progress) / max(self.ramp_fraction, 1e-9))
        values = {}
        if self.terrain_difficulty is not None:
            start, end = self.terrain_difficulty
            values["terrain_difficulty"] = start + (end - start) * ramp
        if self.push_scale is not None:
            start, end = self.push_scale
            values["push_scale"] = start + (end - start) * ramp
        return values


class CurriculumCallback(BaseCallback):
    """Applique le curriculum aux environnements d'entraînement à chaque collecte."""

    def __init__(self, config: CurriculumConfig, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self._config = config
        self._total = total_timesteps

    def _on_rollout_start(self) -> None:
        values = self._config.at(self.num_timesteps / self._total)
        self.training_env.env_method("set_curriculum", **values)
        for key, value in values.items():
            self.logger.record(f"curriculum/{key}", value)

    def _on_step(self) -> bool:
        return True
