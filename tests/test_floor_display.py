"""La couleur du sol suit sa friction, dans le viewer comme dans visualize.py.

Le rendu lui-même demande un écran ; ce qui se teste sans, c'est que la bonne
couleur arrive dans ``geom_rgba`` et que ``make_eval_env`` tire bien un sol
dans la plage du YAML quand on le lui demande.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_yaml, make_poppy_env_config
from src.environments.env_factory import make_eval_env, single_env
from src.environments.floor_display import GRIPPY_RGBA, SLIPPERY_RGBA, color_floor, floor_color
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

_CONFIG = Path(__file__).parent.parent / "configs" / "poppy_robust.yaml"


def test_extremes_de_la_plage_donnent_les_deux_bleus() -> None:
    """Le minimum de friction est bleu clair, le maximum bleu marine."""
    np.testing.assert_allclose(floor_color(0.4, (0.4, 2.0)), SLIPPERY_RGBA)
    np.testing.assert_allclose(floor_color(2.0, (0.4, 2.0)), GRIPPY_RGBA)
    # Hors plage, la couleur reste bornée plutôt que de sortir de [0, 1].
    np.testing.assert_allclose(floor_color(5.0, (0.4, 2.0)), GRIPPY_RGBA)


def test_color_floor_ecrit_la_couleur_du_sol_tire() -> None:
    """Après un reset, le sol prend la couleur de la friction qui vient d'être tirée."""
    env = PoppyHumanoidEnv(floor_noise=True, friction_range=(0.4, 2.0))
    try:
        env.reset(seed=0)
        friction = color_floor(env, (0.4, 2.0))
        floor_id = env.model.geom("floor").id
        np.testing.assert_allclose(
            env.model.geom_rgba[floor_id], floor_color(friction, (0.4, 2.0))
        )
    finally:
        env.close()


def test_make_eval_env_tire_le_sol_dans_la_plage_du_yaml() -> None:
    """Avec floor_noise, la friction suit le YAML ; sans, elle ne bouge pas."""
    config = make_poppy_env_config(load_yaml(_CONFIG))
    low, high = config.domain_randomization.friction_range

    noisy = make_eval_env(config, seed=0, floor_noise=True)
    fixed = make_eval_env(config, seed=0, floor_noise=False)
    try:
        frictions = set()
        for _ in range(5):
            noisy.reset()
            info = single_env(noisy).unwrapped.get_floor_randomization_info()
            frictions.add(info["floor_slide_friction"])
        assert all(low <= f <= high for f in frictions)
        assert len(frictions) > 1, "Le sol devrait changer d'un épisode à l'autre."

        fixed_frictions = set()
        for _ in range(3):
            fixed.reset()
            info = single_env(fixed).unwrapped.get_floor_randomization_info()
            fixed_frictions.add(info["floor_slide_friction"])
        assert len(fixed_frictions) == 1
    finally:
        noisy.close()
        fixed.close()
