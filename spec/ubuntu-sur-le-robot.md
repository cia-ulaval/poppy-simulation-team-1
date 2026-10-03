# Passer le robot sous Ubuntu, sans Docker — proposition

> **Statut : proposition, à discuter en équipe.** Ajoutez vos avis en bas du fichier ou
> en commentaire de l'issue.

## Objectif

Le Raspberry Pi de Poppy tourne sous **Ubuntu 24.04** avec **ROS 2 Jazzy installé
directement**, sans Docker. Le nœud moteurs, l'IMU, l'écran du visage, le son et
rosbridge fonctionnent comme avant.

## Pourquoi

Aujourd'hui, le Pi tourne sous Raspberry Pi OS, où ROS 2 n'est pas supporté. D'où
Docker, qui complique tout : accès au matériel (I2C, série, GPIO), à l'écran, au son,
et une étape de plus pour chaque essai. Sous Ubuntu, ROS 2 s'installe officiellement,
directement.

## À savoir avant de commencer

- **Une réinstallation efface toute la carte SD.** Bonne nouvelle : presque tout ce qui
  tourne sur le robot est déjà sur GitHub (`poppy-conception`), qui sert de sauvegarde.
  Seul ce qui n'a pas été poussé disparaîtrait.
- **Le logiciel Poppy d'origine disparaît aussi**, s'il est sur cette carte : le hotspot
  `10.99.99.1`, la page « Monitor and Control », l'API du port 8080 et les primitives
  (`stand_position`…) décrites dans `Hiver 2026/Doc/Connexion_hotspot.md`. Si l'équipe
  s'en sert encore, il faut le savoir avant.
- **Version de ROS 2 :** le `Dockerfile` actuel utilise *rolling*, une version qui
  change en permanence. Sur Ubuntu 24.04, la version stable à long terme est **Jazzy**.
  Les messages restent compatibles avec le pont de la simulation.
- **Le modèle de Pi décide de la version d'Ubuntu.** Ubuntu 24.04 vise les Pi 4 et 5.
  L'écran du visage a besoin d'un affichage graphique : Ubuntu *Desktop* si le Pi a assez
  de mémoire (4 Go ou plus), sinon Ubuntu *Server* plus un environnement graphique léger.

## Ce qui ne casse pas

- **Le dépôt de simulation** : il ne parle au robot que par rosbridge, à travers le
  réseau. Il faudra seulement lui donner la nouvelle adresse du robot
  (`POPPY_ROSBRIDGE_HOST`).
- **Le code de `poppy-conception`** : c'est du Python, il se recompile en natif avec
  `colcon`.

## À faire

### 1. Avant de toucher à quoi que ce soit

- [ ] Noter le **modèle de Pi** et sa **RAM**.
- [ ] **Utiliser une nouvelle carte SD** pour Ubuntu, et garder l'ancienne intacte. Si
      quelque chose manque, on remet l'ancienne carte et le robot revient exactement
      comme avant.
- [ ] Sur le robot, vérifier qu'il ne reste rien de non poussé : `git status` dans
      `work/poppy-conception`, et repérer les fichiers hors du dépôt (mouvements
      enregistrés, configurations). Le reste est déjà sur GitHub.
- [ ] Décider si le logiciel Poppy d'origine (hotspot, primitives) sert encore.

### 2. Installer Ubuntu

- [ ] Flasher Ubuntu 24.04 (Desktop ou Server, selon la RAM) avec
      [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
- [ ] Créer l'utilisateur, activer SSH, se connecter au réseau.
- [ ] **Choisir de nouveaux mots de passe**, et ne **jamais** les écrire dans le dépôt
      (voir *Sécurité*).

### 3. Installer ROS 2 Jazzy

- [ ] Suivre l'[installation officielle de ROS 2 Jazzy pour Ubuntu](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
      (paquets `.deb`, version `ros-base` suffit).
- [ ] Installer rosbridge et CycloneDDS : `ros-jazzy-rosbridge-server`,
      `ros-jazzy-rmw-cyclonedds-cpp`.
- [ ] Reporter dans `~/.bashrc` ce que faisait le `docker-compose.yml` :
      `ROS_DOMAIN_ID=42`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, et le
      `source /opt/ros/jazzy/setup.bash`.

### 4. Le matériel

- [ ] **Moteurs (USB série) :** ajouter l'utilisateur au groupe `dialout`, installer
      `dynamixel-sdk`.
- [ ] **IMU (MPU6050, I2C) :** activer l'I2C (`dtparam=i2c_arm=on` dans
      `/boot/firmware/config.txt`), installer `i2c-tools`, ajouter l'utilisateur au
      groupe `i2c`. Vérifier : `i2cdetect -y 1` doit montrer l'adresse `68`.
- [ ] **GPIO :** seulement si du code s'en sert ; les bibliothèques diffèrent de
      Raspberry Pi OS, surtout sur Pi 5.
- [ ] **Son :** `arecord -l` et `aplay -l` doivent lister le micro et le haut-parleur.
- [ ] **Écran :** `screen.py` s'ouvre en plein écran (`DISPLAY=:0`).

### 5. Le code

- [ ] Cloner `poppy-conception`, compiler `src/poppy_motors` avec `colcon build`.
- [ ] Lancer rosbridge et vérifier depuis un portable qu'on s'y connecte.
- [ ] **Ne pas lancer le nœud moteurs pour tester** : il active le couple de tous les
      moteurs dès le démarrage (voir *Ne pas faire*). Tester d'abord la lecture seule :
      lire la position d'un moteur.

### 6. Mettre la documentation à jour

- [ ] Réécrire `src/README.md` pour la nouvelle installation, **sans aucun identifiant**.
- [ ] Décider du sort de `Dockerfile` et `docker-compose.yml` : les supprimer, ou les
      garder pour les postes de développement.

## Sécurité

Le dépôt `poppy-conception` est **public**, et deux fichiers contiennent des identifiants
en clair : `src/README.md` (adresse IP, utilisateur et mot de passe SSH) et
`Hiver 2026/Doc/Connexion_hotspot.md` (mot de passe du hotspot).

- [ ] Changer ces mots de passe — la réinstallation est le bon moment.
- [ ] Les retirer des fichiers. Ils restent dans l'**historique git** : les supprimer
      aussi de l'historique, ou considérer les anciens mots de passe comme publics (et
      donc ne plus jamais les utiliser).

## Ne pas faire

- **Effacer l'ancienne carte SD** avant que tout marche sur la nouvelle.
- **Lancer `motor_node.py` sans précaution.** Il active le couple de tous les moteurs au
  démarrage, et sa conversion des angles est provisoire (seuls 3 moteurs inversés, ni
  décalage ni limite). Un essai moteur se fait robot suspendu, quelqu'un la main sur
  l'alimentation.
- **Écrire un mot de passe ou une adresse IP** dans le dépôt.

## Terminé quand

- [ ] Le Pi démarre sous Ubuntu 24.04, ROS 2 Jazzy installé sans Docker.
- [ ] `i2cdetect` voit l'IMU, et ses valeurs se lisent.
- [ ] La position d'un moteur se lit (sans activer le couple).
- [ ] L'écran du visage s'affiche.
- [ ] Le micro et le haut-parleur fonctionnent.
- [ ] rosbridge répond depuis un portable sur le même réseau.
- [ ] `src/README.md` décrit la nouvelle installation, sans identifiant.
- [ ] Les mots de passe ont été changés.
- [ ] L'ancienne carte SD est rangée, étiquetée, intacte.

## Questions pour l'équipe

1. Quel modèle de Raspberry Pi, et combien de RAM ?
2. Le logiciel Poppy d'origine (hotspot, primitives) sert-il encore ?
3. On garde Docker pour les postes de développement, ou on l'enlève complètement ?

## Avis

| Qui | Date | Avis |
| --- | --- | --- |
| | | |
