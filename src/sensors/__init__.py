"""Perception de profondeur. En pause — voir README.md de ce dossier.

Ce fichier n'importe volontairement rien. Il réexportait
``cast_horizontal_rays``, ce qui tirait ``mujoco`` dès qu'on touchait au
paquet : l'image Docker ``vision`` n'a pas MuJoCo, donc un simple
``import src.sensors`` y échouait. Chaque module s'importe directement.
"""
