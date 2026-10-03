# Parler avec Poppy — proposition

> **Statut : proposition, à discuter en équipe.** Ajoutez vos avis en bas du fichier ou
> en commentaire de l'issue.

## Objectif

Pendant une présentation, on pose une question à Poppy à voix haute et il répond avec sa
propre voix, en moins de 2 secondes, pendant que sa bouche bouge à l'écran.

## À savoir avant de commencer

- **Ça tourne sur le robot**, dans le dépôt
  [`poppy-conception`](https://github.com/cia-ulaval/poppy-conception), pas dans la
  simulation.
- **Le micro et le haut-parleur sont déjà branchés et fonctionnels.** Ce n'est pas le
  sujet ici.
- **Le Raspberry Pi est trop peu puissant** pour transcrire et synthétiser la voix
  lui-même. On passe par un service en ligne, donc **il faut Internet** pendant la
  démo, et prévoir quoi faire sans (étape 5).
- **Ça ne touche pas aux moteurs.** Poppy parle, il ne bouge pas, en tout cas dans
  cette tâche.

## Le choix du service de parole

| | **OpenAI Realtime** — recommandé | **ElevenLabs Agents** — alternative |
| --- | --- | --- |
| Principe | Parole → parole en une seule connexion : écoute, réflexion et voix au même endroit | Plateforme complète : voix, base de connaissances intégrée, modèle de langage au choix |
| Modèle | `gpt-realtime-2.1`, ou sa version **mini** pour commencer | Moteur vocal ElevenLabs + un modèle de langage à choisir |
| Prix indicatif | Mini : ~0,016 $/min · standard : ~0,06 à 0,10 $/min | ~0,08 $/min, **plus** le modèle de langage facturé à part ; environ 15 min gratuites par mois |
| Points forts | Le plus simple, le plus rapide. Détecte seul la fin d'une phrase et gère les interruptions. | Les plus belles voix ; on peut créer une **voix de robot sur mesure**. Français supporté. |

*Prix relevés en octobre 2026 sur les pages de prix et des guides publics : à revérifier
au moment de créer les comptes.*

Pour une démo de 20 minutes, les deux coûtent de quelques centimes à quelques dollars :
**le prix ne départagera pas, l'oreille si.** Proposé : commencer avec OpenAI Realtime
mini, et faire écouter ElevenLabs à l'équipe sur les mêmes questions avant de figer le
choix.

## Comment ça s'organise sur le robot

```
micro ──► nœud ROS 2 « voix » ◄──► service de parole (Internet)
              │        │
              │        └──► haut-parleur
              ▼
   sujet ROS 2 « état de la voix » : écoute / réfléchit / parle
              │
              ▼
        écran du visage (src/ui/screen.py) : la bouche bouge quand il parle
```

### Où regarder dans `poppy-conception`

| Fichier | Ce qu'il contient |
| --- | --- |
| `src/ui/screen.py` | Le visage : yeux qui clignent, bouche qui change d'expression. Joue déjà un son avec `aplay` quand on le touche. |
| `src/poppy_motors/` | Un nœud ROS 2 existant : modèle à suivre pour écrire le nœud « voix ». |
| `docker-compose.yml`, `Dockerfile` | L'environnement actuel (ROS 2 dans Docker). Voir la spec `ubuntu-sur-le-robot.md` : il est appelé à changer. |

## La personnalité et la base de connaissances

Un simple fichier texte, que **toute l'équipe** peut enrichir sans programmer. Il est
donné au service de parole comme instructions.

À y mettre :

- **Identité :** son nom, d'où il vient (projet Poppy, club CIA de l'Université Laval),
  83 cm, 25 moteurs.
- **Ce qu'il sait faire, et ce qu'il ne sait pas encore faire.** Par exemple : il
  apprend à marcher en simulation, pas encore en vrai. Important pour qu'il ne s'invente
  pas de capacités devant un public.
- **L'équipe et le projet.**
- **Des choses amusantes :** blagues, anecdotes. Exemple vrai : il a longtemps appris à
  marcher de côté, à cause d'un axe inversé dans sa récompense.

Règles de style :

- Réponses courtes : 1 à 3 phrases.
- Français parlé : pas de listes, pas d'emojis, pas de mise en forme (la voix les lirait).
- « Je ne sais pas » plutôt qu'une invention.

## À faire

1. **Au clavier d'abord.** Un petit script : on tape une question, Poppy répond en texte,
   avec sa personnalité. Sert à mettre au point le fichier de personnalité sans
   s'occuper du son.
2. **À voix haute, en appuyant pour parler.** Brancher le service de parole : on appuie
   sur une touche ou un bouton, on parle, il répond au haut-parleur. Appuyer pour parler
   est plus fiable qu'un mot de réveil dans une salle bruyante.
3. **En nœud ROS 2.** Le nœud « voix » publie son état (écoute / réfléchit / parle).
4. **La bouche bouge.** `screen.py` écoute cet état et anime la bouche quand Poppy parle.
5. **Sans Internet.** Si le service ne répond pas, Poppy le dit avec une phrase
   préenregistrée (« Je n'ai pas de réseau, mais je peux vous faire coucou ! »), au lieu
   de rester muet.
6. **Plus tard :** un mot de réveil (« Hé Poppy »), des gestes en parlant (hocher la
   tête). Les gestes attendent que les moteurs soient pilotés en toute sécurité.

## Ne pas faire

- **Mettre la clé d'API dans le dépôt.** `poppy-conception` est **public** : la clé se
  lit dans une variable d'environnement ou un fichier ignoré par git.
- **Enregistrer l'audio** des conversations. Rien n'est stocké.
- **Commander les moteurs** depuis la voix, dans cette tâche.

## Terminé quand

- [ ] Poppy répond à voix haute à 5 questions de démo préparées.
- [ ] Moins de 2 secondes entre la fin de la question et le début de la réponse.
- [ ] La bouche bouge à l'écran pendant qu'il parle.
- [ ] Sans Internet, il le dit au lieu de rester muet.
- [ ] La personnalité vit dans un fichier texte, modifiable sans toucher au code.
- [ ] Aucune clé d'API dans le dépôt.

## Questions pour l'équipe

1. Quel ton pour Poppy : voix de robot assumée, ou voix naturelle ? Accent québécois ?
2. Qui écrit la personnalité et la base de connaissances ?
3. Qui crée le compte OpenAI (et ElevenLabs pour l'essai), et avec quel budget ?
4. Quelles sont les 5 questions de démo ?

## Avis

| Qui | Date | Avis |
| --- | --- | --- |
| | | |

## Sources (prix)

- [OpenAI — Introducing gpt-realtime](https://openai.com/index/introducing-gpt-realtime/)
- [OpenAI — Advancing voice intelligence with new models in the API](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/)
- [OpenAI Realtime API Pricing 2026 (Layer3 Labs)](https://www.layer3labs.io/guides/openai-realtime-api-pricing)
- [ElevenLabs — API pricing](https://elevenlabs.io/pricing/api)
- [ElevenLabs pricing in 2026 (CloudZero)](https://www.cloudzero.com/blog/elevenlabs-pricing/)
