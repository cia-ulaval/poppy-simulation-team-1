"""Retrouver la bonne normalisation, et seulement elle.

Un modèle chargé avec les statistiques de normalisation d'un *autre* moment de
l'entraînement ne lève aucune erreur : mêmes formes, mêmes types, observations
recentrées sur la mauvaise moyenne. Les chiffres sortis sont faux et rien ne le
dit. C'est le seul mode de défaillance silencieux de l'évaluation, donc le seul
qui mérite un test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate import find_vec_normalize


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_checkpoint_prefere_sa_propre_normalisation(tmp_path: Path) -> None:
    """Un checkpoint intermédiaire ne doit pas attraper le vec_normalize final.

    CheckpointCallback dépose les deux dans le *même* dossier. Prendre le
    générique reviendrait à évaluer le modèle du millionième pas avec les
    statistiques du dernier — l'erreur que ce test existe pour empêcher.
    """
    _touch(
        tmp_path / "poppy_ppo_1000000_steps.zip",
        tmp_path / "poppy_ppo_vecnormalize_1000000_steps.pkl",
        tmp_path / "vec_normalize_final.pkl",
    )

    trouve = find_vec_normalize(tmp_path / "poppy_ppo_1000000_steps.zip", None)

    assert trouve == tmp_path / "poppy_ppo_vecnormalize_1000000_steps.pkl"


def test_modele_final_prend_la_normalisation_finale(tmp_path: Path) -> None:
    """Le modèle de fin d'entraînement, lui, prend bien le fichier générique."""
    _touch(
        tmp_path / "poppy_ppo_final.zip",
        tmp_path / "vec_normalize_final.pkl",
    )

    trouve = find_vec_normalize(tmp_path / "poppy_ppo_final.zip", None)

    assert trouve == tmp_path / "vec_normalize_final.pkl"


def test_best_model_prend_la_normalisation_de_son_dossier(tmp_path: Path) -> None:
    """EvalCallback écrit best_model.zip et vec_normalize.pkl côte à côte.

    C'est la disposition de models/, celle que tout le monde utilisera.
    """
    _touch(
        tmp_path / "best_model.zip",
        tmp_path / "vec_normalize.pkl",
    )

    trouve = find_vec_normalize(tmp_path / "best_model.zip", None)

    assert trouve == tmp_path / "vec_normalize.pkl"


def test_absence_de_normalisation_nest_pas_une_erreur(tmp_path: Path) -> None:
    """Un modèle entraîné sans normalisation est légitime : on renvoie None."""
    _touch(tmp_path / "poppy_ppo_final.zip")

    assert find_vec_normalize(tmp_path / "poppy_ppo_final.zip", None) is None


def test_chemin_explicite_inexistant_echoue_bruyamment(tmp_path: Path) -> None:
    """Une normalisation demandée nommément et introuvable doit lever.

    Retomber silencieusement sur la détection automatique donnerait à
    l'utilisateur d'autres statistiques que celles qu'il a désignées.
    """
    _touch(tmp_path / "poppy_ppo_final.zip", tmp_path / "vec_normalize_final.pkl")

    with pytest.raises(FileNotFoundError):
        find_vec_normalize(
            tmp_path / "poppy_ppo_final.zip", tmp_path / "absent.pkl"
        )
