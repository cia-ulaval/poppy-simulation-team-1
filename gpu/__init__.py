"""Portage GPU de l'entraînement Poppy : mjlab (MuJoCo Warp) et rsl_rl.

Ce paquet ne remplace rien du chemin CPU. Il relit les mêmes YAML de
``configs/`` et la même ``PoppyHumanoidEnv`` pour en tirer les constantes,
puis reproduit pas à pas, en lots sur le GPU, ce que fait ``step()``.
Voir ``gpu/README.md``.
"""
