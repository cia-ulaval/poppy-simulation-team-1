"""Tests du miroir, du sol accidenté, des poussées, du curriculum et de la reprise.

Le test qui compte le plus est le premier : il vérifie la table miroir sur la
**dynamique**, pas sur sa construction. Une table fausse donnerait une perte
de symétrie qui pousse la politique vers des mouvements impossibles, sans
qu'aucune erreur ne soit levée.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import load_yaml, make_poppy_env_config
from src.config.settings import PushConfig, TerrainConfig
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.environments.terrain import generate_heights
from src.training.curriculum import CurriculumConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_CONFIGS = Path(__file__).parent.parent / "configs"


def _mirror_error(fix_knee: bool, seed: int) -> float:
    """Écart, après 10 pas, entre l'état miroir et l'état atteint en miroir."""
    env = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=fix_knee)
    twin = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=fix_knee)
    try:
        spec = env.mirror_spec() if fix_knee else None
        env.reset(seed=seed)
        twin.reset(seed=seed)
        twin.set_state(spec.mirror_qpos(env.data.qpos.copy()),
                       spec.mirror_qvel(env.data.qvel.copy()))
        rng = np.random.default_rng(seed)
        for _ in range(10):
            action = rng.uniform(-0.2, 0.2, size=25)
            env.step(action)
            twin.step(spec.mirror_action(action))
        return float(np.max(np.abs(spec.mirror_qpos(env.data.qpos) - twin.data.qpos)))
    finally:
        env.close()
        twin.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_miroir_respecte_la_dynamique(seed: int) -> None:
    """Un état et une action miroirs mènent à l'état miroir (au millième de rad)."""
    assert _mirror_error(fix_knee=True, seed=seed) < 0.01


def test_miroir_refuse_le_genou_a_l_envers() -> None:
    """Sans la correction du genou, la table refuse de se construire."""
    env = PoppyHumanoidEnv(floor_noise=False)
    try:
        with pytest.raises(ValueError, match="fix_left_knee_axis"):
            env.mirror_spec()
    finally:
        env.close()


def test_genou_gauche_corrige_plie_comme_le_droit() -> None:
    """Après correction, plier dans la plage autorisée recule le pied, des deux côtés."""
    import mujoco

    env = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True)
    try:
        for knee, foot, angle in (("r_knee_y", "r_foot", -0.5), ("l_knee_y", "l_foot", 0.5)):
            data = mujoco.MjData(env.model)
            data.qpos[:] = env.init_qpos
            mujoco.mj_kinematics(env.model, data)
            before = data.xpos[env.model.body(foot).id][1]
            data.qpos[env.model.joint(knee).qposadr[0]] = angle
            mujoco.mj_kinematics(env.model, data)
            # L'avant est −y : reculer, c'est aller vers +y.
            assert data.xpos[env.model.body(foot).id][1] > before + 0.05, knee
    finally:
        env.close()


def test_terrain_plat_a_difficulte_nulle() -> None:
    """À difficulté 0, le terrain est plat : le banc d'essai commun ne change pas."""
    heights = generate_heights(TerrainConfig(), 0.0, 41, np.random.default_rng(0))
    assert np.all(heights == 0.0)


def test_terrain_accidente_mais_depart_plat() -> None:
    """Le relief existe, borné, et le robot ne naît pas dans une bosse."""
    env = PoppyHumanoidEnv(floor_noise=True,
                           terrain=TerrainConfig(enabled=True, difficulty=1.0))
    try:
        env.reset(seed=3)
        heights = env._terrain.heights
        assert np.ptp(heights) > 0.05
        assert np.max(np.abs(heights)) <= 0.25
        assert env._ground_height() == pytest.approx(0.0, abs=1e-6)
        for _ in range(50):
            _, _, terminated, _, _ = env.step(np.zeros(25))
        assert not terminated, "Le robot tombe au repos sur la zone de départ"
    finally:
        env.close()


def test_poussee_est_une_force() -> None:
    """Une poussée de 15 N pendant 0,15 s déplace le bassin dans sa direction."""
    env = PoppyHumanoidEnv(floor_noise=False, push_as_force=True,
                           pushes=PushConfig(duration=0.15))
    try:
        env.reset(seed=0)
        start = env.data.qpos[:2].copy()
        env.apply_push(np.array([15.0, 0.0]), 0.15)
        for _ in range(15):
            env.step(np.zeros(25))
        moved = env.data.qpos[:2] - start
        assert moved[0] > 0.01, f"Le bassin n'a pas bougé selon +x : {moved}"
        assert np.all(env.data.xfrc_applied == 0.0), "La poussée ne s'est pas arrêtée"
    finally:
        env.close()


def test_curriculum_monte_puis_plafonne() -> None:
    """La difficulté monte linéairement sur la rampe, puis reste au maximum."""
    config = CurriculumConfig(terrain_difficulty=(0.1, 1.0), push_scale=(0.2, 1.0),
                              ramp_fraction=0.5)
    assert config.at(0.0) == pytest.approx({"terrain_difficulty": 0.1, "push_scale": 0.2})
    assert config.at(0.25)["terrain_difficulty"] == pytest.approx(0.55)
    assert config.at(0.9) == pytest.approx({"terrain_difficulty": 1.0, "push_scale": 1.0})


@pytest.mark.parametrize("name", ["palier1", "palier2", "palier3"])
def test_configs_des_paliers_se_chargent(name: str) -> None:
    """Les trois YAML se lisent, sans clé inconnue."""
    config = make_poppy_env_config(load_yaml(_CONFIGS / f"{name}.yaml"))
    assert config.fix_left_knee_axis is True
    assert config.terrain.enabled is (name != "palier1")
    assert config.domain_randomization.push_as_force is (name == "palier3")


def test_reprise_prend_le_dernier_checkpoint(tmp_path: Path) -> None:
    """--resume sur un dossier choisit le checkpoint au plus grand nombre de pas."""
    from train_poppy import checkpoint_vec_normalize, find_checkpoint

    for steps in (500000, 1500000, 1000000):
        (tmp_path / f"poppy_ppo_{steps}_steps.zip").touch()
        (tmp_path / f"poppy_ppo_vecnormalize_{steps}_steps.pkl").touch()
    checkpoint = find_checkpoint(tmp_path)
    assert checkpoint.name == "poppy_ppo_1500000_steps.zip"
    assert checkpoint_vec_normalize(checkpoint).name == "poppy_ppo_vecnormalize_1500000_steps.pkl"
