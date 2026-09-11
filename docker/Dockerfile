# syntax=docker/dockerfile:1
#
# Images du projet Poppy Simulation — un seul fichier, plusieurs cibles.
#
# Un Dockerfile multi-étages plutôt que quatre fichiers séparés : les
# dépendances communes sont déclarées une seule fois (étage `base`) et les
# trois services en héritent. Pas de dérive entre images, pas d'ordre de
# build à orchestrer à la main.
#
# Cibles :
#   base       socle Python + torch + Stable-Baselines3 (jamais lancé seul)
#   train      entraînement RL, MuJoCo hors écran + TensorBoard
#   bridge     pont vers le robot réel via rosbridge (roslibpy)
#   vision     serveur d'estimation de profondeur (websocket)
#   rosbridge  faux robot : un rosbridge en conteneur pour tester sans robot
#
# Dépendances : chaque cible installe requirements/<cible>.${REQUIREMENTS}.
#   txt  (défaut) plages de versions, tolérant, compatible GPU
#   lock versions exactes relevées dans une image construite
# REQUIREMENTS=lock docker compose build train  pour une image reproductible.
#
# Build : docker compose build   (voir docs/DOCKER.md)


# ---------------------------------------------------------------------------
# base — socle commun
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Index PyPI de torch. CPU par défaut : ça marche sur toutes les machines de
# l'équipe sans pilote NVIDIA. Pour construire la variante GPU :
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 docker compose build train
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements/ /tmp/requirements/

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX}" torch \
    && python -m pip install -r /tmp/requirements/base.txt

# Utilisateur non privilégié : rien dans ces conteneurs n'a besoin de root.
RUN useradd --create-home --uid 1000 poppy \
    && chown poppy:poppy /workspace


# ---------------------------------------------------------------------------
# train — entraînement par renforcement
# ---------------------------------------------------------------------------
FROM base AS train

# libosmesa6 : rendu MuJoCo hors écran, nécessaire uniquement pour enregistrer
# des vidéos d'épisodes. L'entraînement pur (render_mode=None) n'en a pas besoin.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# txt par défaut : voir l'en-tête du fichier. Le .lock contient la fermeture
# transitive complète, torch compris ; ce que base a déjà installé y est
# simplement épinglé à la version exacte, sans nouveau téléchargement.
ARG REQUIREMENTS=txt
RUN python -m pip install -r "/tmp/requirements/train.${REQUIREMENTS}"

# pytest et ruff : l'image train sert aussi d'image de développement, étant la
# seule à posséder MuJoCo. Une vingtaine de Mo, pour que `docker compose run
# --rm train pytest` fonctionne sans seconde image à construire.
RUN python -m pip install -r /tmp/requirements/dev-tools.txt

# osmesa = rendu logiciel : fonctionne sans GPU ni serveur X.
ENV MUJOCO_GL=osmesa

COPY --chown=poppy:poppy src/ /workspace/src/
COPY --chown=poppy:poppy scripts/ /workspace/scripts/
COPY --chown=poppy:poppy configs/ /workspace/configs/
COPY --chown=poppy:poppy assets/ /workspace/assets/
# tests/ et pyproject.toml : l'image doit pouvoir lancer pytest seule, sans
# dépendre des volumes montés par compose.
COPY --chown=poppy:poppy tests/ /workspace/tests/
COPY --chown=poppy:poppy pyproject.toml /workspace/pyproject.toml

USER poppy
CMD ["python", "scripts/train_poppy.py", "--help"]


# ---------------------------------------------------------------------------
# bridge — pont vers le robot réel
# ---------------------------------------------------------------------------
FROM base AS bridge

# txt par défaut : voir l'en-tête du fichier. Le .lock contient la fermeture
# transitive complète, torch compris ; ce que base a déjà installé y est
# simplement épinglé à la version exacte, sans nouveau téléchargement.
ARG REQUIREMENTS=txt
RUN python -m pip install -r "/tmp/requirements/bridge.${REQUIREMENTS}"

COPY --chown=poppy:poppy src/ /workspace/src/
COPY --chown=poppy:poppy scripts/ /workspace/scripts/
COPY --chown=poppy:poppy configs/ /workspace/configs/
COPY --chown=poppy:poppy assets/ /workspace/assets/

# Adresse du rosbridge, lue par src/robot/ros_publisher.py. Le défaut vise le
# faux robot de la stack compose — jamais un robot réel, qu'on ne doit
# atteindre qu'en fournissant explicitement son adresse.
ENV POPPY_ROSBRIDGE_HOST=rosbridge \
    POPPY_ROSBRIDGE_PORT=9090

USER poppy
CMD ["python", "scripts/run_robot.py", "--help"]


# ---------------------------------------------------------------------------
# vision — serveur de profondeur
# ---------------------------------------------------------------------------
FROM base AS vision

# txt par défaut : voir l'en-tête du fichier. Le .lock contient la fermeture
# transitive complète, torch compris ; ce que base a déjà installé y est
# simplement épinglé à la version exacte, sans nouveau téléchargement.
ARG REQUIREMENTS=txt
RUN python -m pip install -r "/tmp/requirements/vision.${REQUIREMENTS}"

COPY --chown=poppy:poppy src/ /workspace/src/

EXPOSE 8000
USER poppy
# Lancé par chemin de fichier, pas par `python -m`. Passer par le module
# importerait le paquet `src`, dont les __init__ tirent mujoco, gymnasium et
# matplotlib — absents de cette image, et sans rapport avec la profondeur.
CMD ["python", "src/sensors/depth_server.py"]


# ---------------------------------------------------------------------------
# rosbridge — faux robot, pour tester le pont sans matériel
# ---------------------------------------------------------------------------
# Seul étage à ne pas hériter de `base` : il lui faut un vrai ROS 2.
# Il ne simule aucune physique — il accepte les connexions et relaie les
# messages, ce qui suffit à vérifier que le pont publie ce qu'on croit.
FROM ros:humble-ros-base AS rosbridge

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 9090
CMD ["bash", "-lc", "source /opt/ros/humble/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml"]
