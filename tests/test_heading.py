"""Ce que « avancer » veut dire pour ce robot.

Ce test existe parce que l'erreur qu'il empêche a coûté cher : la récompense
mesurait l'avance sur l'axe x du monde, qui est l'axe **gauche-droite** du
Poppy. Elle récompensait donc le pas chassé et facturait la vraie marche au
titre de la dérive latérale, sans que rien ne le signale. Le classement des
onze modèles entraînés en était inversé.

Rien ici ne dépend d'une politique entraînée : uniquement de la géométrie du
modèle et du repère du bassin.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv


@pytest.fixture
def env() -> PoppyHumanoidEnv:
    """Environnement déterministe : sol fixe, aucun rendu."""
    environment = PoppyHumanoidEnv(floor_noise=False, render_mode=None)
    environment.reset(seed=0)
    yield environment
    environment.close()


def test_axe_sagittal_est_y(env: PoppyHumanoidEnv) -> None:
    """Les pieds sont côte à côte selon x, donc le robot avance selon y.

    C'est la mesure d'où tout découle. Si elle change, c'est que le modèle
    MJCF a été régénéré avec une autre convention — et alors la récompense,
    les caméras et le pont robot sont tous à revoir.
    """
    model, data = env.model, env.data
    gauche = data.xpos[model.body("l_foot").id]
    droit = data.xpos[model.body("r_foot").id]
    ecart = gauche - droit

    assert abs(ecart[0]) > 5 * abs(ecart[1]), (
        f"Les pieds devraient être séparés selon x (écart {ecart[:2]}). "
        f"S'ils le sont selon y, l'axe d'avance du robot a changé."
    )


def test_avant_du_robot_est_moins_y(env: PoppyHumanoidEnv) -> None:
    """Au repos, l'avant du robot pointe vers -y et sa gauche vers +x.

    Le sens vient de la flexion du genou : plier un genou envoie le pied vers
    l'arrière et vers le haut. Sur la jambe droite, dont la course est
    [-2.34, +0.06] rad, la flexion est négative et envoie le pied en +y —
    donc +y est l'arrière.
    """
    avant, gauche = env._heading_frame()

    np.testing.assert_allclose(avant, [0.0, -1.0], atol=0.02)
    np.testing.assert_allclose(gauche, [1.0, 0.0], atol=0.02)


def test_le_cap_suit_la_rotation_du_robot(env: PoppyHumanoidEnv) -> None:
    """Tourné de 90°, le robot doit avoir un « avant » tourné de 90°.

    Sans cela, la randomisation du cap initial (±15° à chaque reset) fausserait
    le signal d'avance : une politique serait récompensée différemment selon
    l'orientation de départ, pour un comportement identique.
    """
    qpos = env.init_qpos.copy()
    demi = np.pi / 4  # rotation de +90° autour de z
    qpos[3:7] = [np.cos(demi), 0.0, 0.0, np.sin(demi)]
    env.set_state(qpos, env.init_qvel.copy())

    avant, _ = env._heading_frame()

    np.testing.assert_allclose(avant, [1.0, 0.0], atol=0.02)


def test_la_recompense_credite_l_avance_du_robot(env: PoppyHumanoidEnv) -> None:
    """Un déplacement vers l'avant du robot compte comme de l'avance, pas de la dérive.

    C'est le contrat que la version précédente violait : elle projetait sur un
    axe fixe du monde, donc un robot marchant parfaitement droit mais orienté
    autrement était compté comme dérivant.
    """
    avant, gauche = env._heading_frame()
    vitesse_avant = 0.5 * avant  # 0,5 m/s droit devant

    avance = float(vitesse_avant @ avant)
    derive = float(vitesse_avant @ gauche)

    assert avance == pytest.approx(0.5, abs=1e-6)
    # Les deux axes sont projetés au sol puis normalisés séparément : ils ne
    # sont donc pas exactement orthogonaux dès que le bassin a un peu de
    # roulis ou de tangage, ce que le bruit de reset (±0,01 rad) suffit à
    # produire. Le contrat utile n'est pas l'orthogonalité parfaite, c'est que
    # la dérive soit négligeable devant l'avance.
    assert abs(derive) < 1e-3 * abs(avance)


def test_un_pas_renvoie_une_avance_coherente(env: PoppyHumanoidEnv) -> None:
    """``forward_vel`` de l'info doit être la projection sur le cap, pas sur x.

    On compare la valeur rapportée au déplacement réellement mesuré dans la
    simulation, projeté à la main.
    """
    avant_pos = env.data.qpos[:2].copy()
    avant_dir, _ = env._heading_frame()

    rng = np.random.default_rng(0)
    action = rng.uniform(-1.0, 1.0, size=env.action_space.shape[0]).astype(np.float32)
    _, _, _, _, info = env.step(action)

    deplacement = env.data.qpos[:2].copy() - avant_pos
    attendu = float(deplacement @ avant_dir) / env.dt

    assert info["forward_vel"] == pytest.approx(attendu, rel=1e-3, abs=1e-6)
