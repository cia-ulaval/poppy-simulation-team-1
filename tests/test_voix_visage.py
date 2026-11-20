"""Contrat entre la voix (voice_assistant/) et le visage (src/ui/screen.py).

screen.py ouvre une fenêtre dès son import, et voice_assistant/ demande le son et la
clé : on lit donc screen.py sans l'exécuter, et on ne charge que visage.py, qui ne
dépend que de la bibliothèque standard.
"""

import ast
import importlib.util
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SCREEN = RACINE / "src" / "ui" / "screen.py"


def _charger_visage():
    spec = importlib.util.spec_from_file_location(
        "visage_voix", RACINE / "voice_assistant" / "visage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emotions_du_visage():
    for noeud in ast.parse(SCREEN.read_text(encoding="utf-8")).body:
        if (isinstance(noeud, ast.Assign)
                and any(getattr(cible, "id", None) == "EMOTIONS" for cible in noeud.targets)):
            return tuple(ast.literal_eval(cle) for cle in noeud.value.keys)
    raise AssertionError("EMOTIONS introuvable dans screen.py")


def test_gpt_ne_peut_choisir_qu_un_visage_qui_existe():
    assert _charger_visage().EMOTIONS_VISAGE == _emotions_du_visage()


def test_meme_fichier_des_deux_cotes():
    source = SCREEN.read_text(encoding="utf-8")
    assert '"poppy_visage"' in source and 'Path("/dev/shm")' in source
    assert _charger_visage().chemin_visage().name == "poppy_visage"


def test_ecriture_complete_et_emotion_inconnue_ignoree(tmp_path):
    visage = _charger_visage().Visage("happy", chemin=tmp_path / "poppy_visage")
    assert visage.changer_emotion("sad")
    assert not visage.changer_emotion("talking")  # la parole n'est pas une émotion
    visage.bouche_ouverte(True, 1.7)
    visage.ecrire()
    etat = json.loads((tmp_path / "poppy_visage").read_text(encoding="utf-8"))
    assert etat == {"emotion": "sad", "parle": True, "bouche": 1.0}
    assert list(tmp_path.iterdir()) == [tmp_path / "poppy_visage"]  # pas de .tmp laissé
    visage.arreter()
    assert not (tmp_path / "poppy_visage").exists()
