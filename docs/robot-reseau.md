# Se connecter au Poppy réel, sur n'importe quel réseau

État au 20 novembre 2026. Le robot se reconnecte tout seul, envoie son adresse sur
le téléphone de l'équipe, et crée son propre Wi-Fi s'il ne trouve rien. Ce document
dit comment le joindre en SSH, à l'université comme ailleurs.

Le **mot de passe** du Wi-Fi du robot et le **sujet ntfy** ne sont pas écrits ici :
le dépôt est public. Ils sont dans le message épinglé du Discord de l'équipe.

## En bref

| Le robot est sur… | Ton ordinateur doit être sur… | Commande |
|---|---|---|
| eduroam (université) | eduroam | `ssh poppy@10.242.180.129` |
| un partage de connexion ou un autre Wi-Fi | le même réseau | l'adresse reçue sur ntfy, ou `ssh poppy@poppy.local` |
| son propre Wi-Fi « Poppy » | le Wi-Fi « Poppy » | `ssh poppy@10.42.0.1` |

L'adresse eduroam est celle qu'eduroam lui donne en ce moment ; elle peut changer.
Celle du Wi-Fi « Poppy » ne change jamais.

## Ce que fait le robot quand il n'a plus de Wi-Fi

Le service `poppy-wifi-secours` tourne en permanence sur le robot.

1. Il essaie les **réseaux qu'il connaît déjà** : eduroam, partages de connexion
   déjà utilisés, etc.
2. Sinon, il essaie les **réseaux visibles protégés par mot de passe**, les six plus
   forts, avec le **mot de passe par défaut de l'équipe** (voir Discord). Un réseau
   qui accepte est gardé : il devient un réseau connu, sous le nom `auto-<nom>`.
3. Sinon, après trois tours, environ 5 minutes, il crée son **Wi-Fi « Poppy »**, à
   l'adresse fixe `10.42.0.1`.
4. Sur « Poppy », si personne n'est connecté depuis 5 minutes, il recommence à
   l'étape 1. Il revient donc tout seul sur eduroam dès qu'il le voit.

**Pour une présentation hors université**, le plus simple est donc d'activer le
partage de connexion d'un téléphone avec le mot de passe par défaut, sous n'importe
quel nom. Le robot s'y connecte seul et envoie son adresse.

## Connaître l'adresse du robot : ntfy

À chaque connexion à un réseau qui a internet, le robot envoie une notification :

> **Poppy est connecté**
> Réseau : auto-iPhone de …
> ssh poppy@172.20.10.2

Pour la recevoir, une seule fois :

1. installer l'appli **ntfy** (App Store ou Play Store, gratuite, sans compte) ;
2. **+** → « S'abonner à un sujet » → le sujet donné sur Discord.

Sans l'appli, on peut aussi ouvrir `https://ntfy.sh/<sujet>` dans un navigateur.

Sur son propre Wi-Fi « Poppy », il n'y a pas d'internet, donc pas de notification :
c'est toujours `10.42.0.1`.

Si la notification n'arrive pas, `ssh poppy@poppy.local` marche sur la plupart des
partages de connexion et des Wi-Fi de maison. Il ne marche pas sur eduroam.

## Se connecter au Wi-Fi « Poppy »

1. Choisir **Poppy** dans la liste des Wi-Fi.
2. Windows demande d'abord un **code NIP** : cliquer sur **« Se connecter à
   l'aide d'un mot de passe à la place »**, puis entrer le mot de passe (Discord).
3. `ssh poppy@10.42.0.1`

Windows quitte parfois « Poppy », qui n'a pas internet, pour revenir sur un réseau
connu comme eduroam. Pendant une présentation, cocher « Se connecter
automatiquement » sur « Poppy ».

Pour forcer « Poppy » à la main, depuis le robot (la session SSH par Wi-Fi sera
coupée) :

```bash
sudo nmcli con up poppy-ap
```

## Ajouter ou retirer un réseau connu

À faire **pendant qu'on est encore connecté** au robot :

```bash
sudo nmcli con add type wifi con-name "<nom>" ssid "<nom exact du Wi-Fi>" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<mot de passe>" connection.autoconnect yes
```

```bash
nmcli -t -f NAME,TYPE con show | grep wireless
```

```bash
sudo nmcli con delete "<nom>"
```

Un réseau connu est **prioritaire** sur l'essai du mot de passe par défaut. S'il est
visible, le robot s'y connecte d'abord.

## Tester sans eduroam

À l'université, eduroam est toujours visible, donc le robot n'essaie jamais rien
d'autre. Pour tester le reste, on lui cache eduroam :

```bash
sudo systemd-run --unit=poppy-eduroam-secours --on-active=60min --collect /usr/local/bin/poppy-eduroam remettre
```

```bash
sudo /usr/local/bin/poppy-eduroam pause
```

```bash
sudo systemd-run --collect sh -c "sleep 3; nmcli con down eduroam2"
```

La première commande est un **filet de sécurité** : eduroam revient tout seul au bout
de 60 minutes, même si on a perdu le contact avec le robot.

Pour rendre eduroam avant, depuis le réseau où se trouve le robot :

```bash
sudo /usr/local/bin/poppy-eduroam remettre
```

```bash
sudo systemd-run --collect sh -c "sleep 2; nmcli con up eduroam2"
```

La session SSH se coupe au bout de 2 secondes : c'est normal. Il faut ensuite remettre
son ordinateur sur eduroam.

## Dépannage

| Symptôme | Vérifier |
|---|---|
| Aucune notification | Le réseau a-t-il internet ? Le robot est-il peut-être sur « Poppy » ? |
| « Poppy » n'apparaît pas | Le robot essaie encore des réseaux : attendre 5 minutes. |
| `ssh` ne répond pas | L'ordinateur est-il sur **le même réseau** que le robot ? |
| `Host key verification failed` sur une nouvelle adresse | C'est le même robot à une autre adresse : `ssh -o HostKeyAlias=10.242.180.129 poppy@<adresse>` |

Le journal du service, pour comprendre ce qu'il a essayé :

```bash
journalctl -t poppy-wifi-secours -t poppy-eduroam --since "-1h" --no-pager
```

## Ce qui est installé sur le robot

| Fichier | Rôle |
|---|---|
| `/usr/local/bin/poppy-wifi-secours` | la boucle décrite plus haut |
| `/etc/systemd/system/poppy-wifi-secours.service` | la lance au démarrage et la relance si elle s'arrête |
| connexion NetworkManager `poppy-ap` | le Wi-Fi « Poppy » (SSID `Poppy`, `10.42.0.1/24`, WPA2) |
| `/etc/NetworkManager/dispatcher.d/90-poppy-ntfy` | envoie la notification à chaque connexion |
| `/etc/poppy/ntfy.conf` | le sujet ntfy |
| `/usr/local/bin/poppy-eduroam` | `pause` / `remettre` pour les tests |

Rien de tout cela n'est dans `/tmp` : tout survit au redémarrage.
