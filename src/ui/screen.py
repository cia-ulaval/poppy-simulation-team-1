import json
import math
import random
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

root = tk.Tk()
if "--fenetre" in sys.argv:  # sur un ordinateur : une fenêtre à la taille de l'écran du robot
    root.geometry("800x480")
    root.title("Poppy")
else:
    root.attributes("-fullscreen", True)
root.configure(bg="black")

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Pour les sourcils (ils n'apparaissent que pour certaines émotions)
left_brow = canvas.create_line(0, 0, 0, 0, fill="white", width=6, state="hidden")
right_brow = canvas.create_line(0, 0, 0, 0, fill="white", width=6, state="hidden")

# Pour les joues (elles n'apparaissent que pour certaines émotions)
left_cheek = canvas.create_oval(120, 270, 180, 294, fill="#ff7aa2", outline="", state="hidden")
right_cheek = canvas.create_oval(620, 270, 680, 294, fill="#ff7aa2", outline="", state="hidden")

# Pour les larmes (elles n'apparaissent que pour certaines émotions)
tear = canvas.create_oval(175, 255, 195, 285, fill="#5fb4ff", outline="", state="hidden")

thinking_point1 = canvas.create_oval(473, 330, 487, 344, fill="#ffd34d", outline="", state="hidden")
thinking_point2 = canvas.create_oval(498, 330, 512, 344, fill="#ffd34d", outline="", state="hidden")
thinking_point3 = canvas.create_oval(523, 330, 537, 344, fill="#ffd34d", outline="", state="hidden")

# pour les yeux
left_eye = canvas.create_oval(150, 150, 250, 250, fill="white")
right_eye = canvas.create_oval(550, 150, 650, 250, fill="white")

# pour les pupille
left_pupil = canvas.create_oval(180, 180, 220, 220, fill="black")
right_pupil = canvas.create_oval(580, 180, 620, 220, fill="black")

# Pour le visage 67 : un « 6 » et un « 7 » à la place des yeux (cachés par défaut)
DIGIT_YELLOW, DIGIT_PALE = "#ffd34d", "#9c8434"
DIGIT_SIZE = 160  # hauteur de la police des chiffres, en px
left_digit = canvas.create_text(200, 195, text="6", fill=DIGIT_YELLOW,
                                font=("Helvetica", -DIGIT_SIZE, "bold"), state="hidden")
right_digit = canvas.create_text(600, 195, text="7", fill=DIGIT_YELLOW,
                                 font=("Helvetica", -DIGIT_SIZE, "bold"), state="hidden")

# pour la bouche
mouth_closed = canvas.create_arc(300, 280, 500, 400, start=0, extent=-180, style="arc", width=4, outline="white")
mouth_open = canvas.create_oval(0, 0, 0, 0, fill="white", outline="", state="hidden")

EXTRAS = {
    "cheeks": [left_cheek, right_cheek],
    "tear": [tear],
    "thinking_points": [thinking_point1, thinking_point2, thinking_point3],
    "digits": [left_digit, right_digit],
}

EMOTIONS = {
    "happy": {
        # (x1, y1, x2, y2) : boîte dans laquelle la bouche est dessinée
        "mouth_box": (300, 280, 500, 400),
        "mouth_start": 0,
        "mouth_extent": -180,
        "eye_color": "white",
        "brows": None,
        "pupil_offset": (0, -8),  # décalage de 8px vers le haut des pupilles
        "extras": ["cheeks"],
    },
    "sad": {
        "mouth_box": (320, 330, 480, 410),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "#bcd4ff",
        "pupil_offset": (0, 15),
        "brows": ((150, 140, 250, 118), (550, 118, 650, 140)),
        "extras": ["tear"],
    },
    "thinking": {
        "mouth_box": (350, 339, 440, 341),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "white",
        "pupil_offset": (15, -15),
        "brows": ((150, 130, 250, 130), (550, 118, 650, 100)),
        "extras": ["thinking_points"],
    },
    "neutral" : {
        "mouth_box": (350, 339, 440, 341),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "white",
        "brows" : None,
        "pupil_offset" : (0, 0),
        "extras": [],
    },
    "angry" : {
        "mouth_box": (320, 310, 480, 370),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "#e24b4a",
        "pupil_offset" : (0, 6),
        "brows" : ((140, 120, 260, 165), (540, 165, 660, 120)),
        "extras": [],
    },
    "67" : {
        "mouth_box": (288, 270, 512, 430),
        "mouth_start": 0,
        "mouth_extent": -180,
        "eye_color": "white",
        "round_eyes": False,  # les chiffres remplacent les yeux et les pupilles
        "pupil_offset" : (0, 0),
        "brows" : ((158, 94, 242, 114), (558, 114, 642, 94)),
        "extras": ["cheeks", "digits"],
    },
}

ANGRY_SHAKE = 5

# Visage 67 : les chiffres balancent au rythme de la danse. scripts/robot_danse.py écrit
# « 6 » ou « 7 » dans ce fichier à chaque battement ; il est facultatif (absent sous
# Windows, ignoré s'il a plus de 3 s) et, sans lui, le visage garde sa propre cadence.
CADENCE_67 = Path("/tmp/poppy_67")
CADENCE_67_FRESH_S = 3.0
CADENCE_67_FREE_MS = 650  # un battement sans cadence : à peu près celui du robot
DIGIT_SWAY = 24  # course du balancement des chiffres, en px

# La parole n'est pas une émotion : elle ne prend que la bouche (et lève un peu les
# sourcils), par-dessus l'émotion en cours, qui garde ses yeux, ses sourcils et ses
# extras. On peut donc parler en étant triste, content, en colère...
TALK_CENTER = (400, 342)
TALK_BROW_LIFT = 12  # montée des sourcils quand la bouche est grande ouverte, en px

MOUTH_SHAPES = {
    "close": {"width": 120, "height": 10, "weight": 3},
    "e":     {"width": 150, "height": 28, "weight": 2},
    "a":     {"width": 110, "height": 80, "weight": 3},
    "o":     {"width": 60,  "height": 64, "weight": 2},
    "u":     {"width": 44,  "height": 30, "weight": 1},
    "i":     {"width": 140, "height": 14, "weight": 2},
}

# La voix (voice_assistant/) pilote le visage par un petit fichier JSON, relu toutes les
# 40 ms : {"emotion": "happy", "parle": true, "bouche": 0.6}. Même emplacement que dans
# voice_assistant/visage.py : /dev/shm sur le robot (en mémoire, pour ne pas user la carte
# SD), sinon le dossier temporaire. Fichier absent, illisible ou vieux de plus de 3 s : le
# visage vit sa vie, comme sans la voix.
VOIX = (Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())) / "poppy_visage"
VOIX_FRESH_S = 3.0
VOIX_PERIOD_MS = 40
voix_emotion = None  # dernière émotion lue dans le fichier
voix_parle = False  # dernier « parle » lu dans le fichier
voix_bouche = None  # ouverture de la bouche voulue par la voix, ou None : au hasard

talk_width, talk_height = 120, 10
talk_target = MOUTH_SHAPES["close"]

shake_sign = 1
current_emotion = "happy"
emotion_cycle = list(EMOTIONS)
animation_job = None  # animation de l'émotion : larme, points, colère

talking = False
talk_job = None  # animation de la bouche qui parle, indépendante de celle de l'émotion
brow_shake = (0, 0)  # décalage des sourcils dû à la colère
brow_lift = 0  # montée des sourcils due à la parole

digit_phase = "6"  # chiffre du battement en cours
digit_x = 0.0  # balancement, de -1 (gauche de l'écran) à 1 (droite)
digit_beat_start = 0.0  # instant du dernier battement
digit_last_signal = 0.0  # dernier instant où le fichier de cadence a été lu
digit_blink_until = 0.0  # les chiffres disparaissent brièvement jusque-là

def set_emotion(emotion):
    global current_emotion
    global animation_job
    global brow_shake
    global digit_phase, digit_x, digit_beat_start, digit_last_signal

    current_emotion = emotion

    fiche = EMOTIONS[emotion]

    for extra in EXTRAS:
        for item in EXTRAS[extra]:
            canvas.itemconfig(item, state="hidden")
    for extra in fiche["extras"]:
        for item in EXTRAS[extra]:
            canvas.itemconfig(item, state="normal")

    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

    # Les yeux ronds et leurs pupilles disparaissent quand des chiffres les remplacent.
    eyes_state = "normal" if fiche.get("round_eyes", True) else "hidden"
    for item in (left_eye, right_eye, left_pupil, right_pupil):
        canvas.itemconfig(item, state=eyes_state)

    # Pendant la parole, la bouche appartient à la parole : on n'y touche pas.
    if not talking:
        draw_rest_mouth()

    dx, dy = fiche["pupil_offset"]
    place_pupils(dx, dy)

    brow_shake = (0, 0)
    place_brows()

    # ANIMATIONS
    if animation_job is not None:
        root.after_cancel(animation_job)
        animation_job = None
    canvas.coords(tear, 175, 255, 195, 285)
    if emotion == "sad":
        animate_tear()
    elif emotion == "thinking":
        animate_thinking_points(0)
    elif emotion == "angry":
        animate_angry()
    elif emotion == "67":
        digit_phase, digit_x = "6", 0.0
        digit_beat_start = time.monotonic()
        digit_last_signal = -CADENCE_67_FRESH_S
        animate_67()

def draw_rest_mouth():
    """Bouche de l'émotion en cours, quand Poppy ne parle pas."""
    fiche = EMOTIONS[current_emotion]
    canvas.coords(mouth_closed, *fiche["mouth_box"])
    canvas.itemconfig(mouth_closed, start=fiche["mouth_start"], extent=fiche["mouth_extent"],
                      style="arc", fill="", outline="white", width=4, state="normal")
    canvas.itemconfig(mouth_open, state="hidden")

def place_brows():
    """Sourcils de l'émotion en cours, secoués par la colère et levés par la parole."""
    brows = EMOTIONS[current_emotion]["brows"]
    if brows is None:  # émotion sans sourcils : ils restent cachés, même en parlant
        canvas.itemconfig(left_brow, state="hidden")
        canvas.itemconfig(right_brow, state="hidden")
        return

    dx, dy = brow_shake
    for brow, coords in ((left_brow, brows[0]), (right_brow, brows[1])):
        canvas.coords(brow, *coords)
        canvas.move(brow, dx, dy + brow_lift)
        canvas.itemconfig(brow, state="normal")

def set_talking(active):
    """Fait parler Poppy, ou le fait taire, sans changer son émotion."""
    global talking, talk_job, brow_lift, talk_width, talk_height, talk_target

    if active == talking:
        return
    talking = active

    if talk_job is not None:
        root.after_cancel(talk_job)
        talk_job = None

    if talking:
        animate_talking()
    else:
        talk_width, talk_height = 120, 10
        talk_target = MOUTH_SHAPES["close"]
        brow_lift = 0
        place_brows()
        draw_rest_mouth()

def toggle_talking(event):
    set_talking(not talking)

def next_emotion(event):
    next_emotion_index = (emotion_cycle.index(current_emotion) + 1) % len(emotion_cycle)
    set_emotion(emotion_cycle[next_emotion_index])

def animate_tear():
    global animation_job
    canvas.move(tear, 0, 2)

    if canvas.coords(tear)[1] > 370:
        canvas.coords(tear, 175, 255, 195, 285)

    animation_job = root.after(50, animate_tear)

def yellow_at(level):
    """Jaune des points à la luminosité level : 0 = noir (invisible), 1 = #ffd34d."""
    red = round(255 * level)
    green = round(211 * level)
    blue = round(77 * level)
    return f"#{red:02x}{green:02x}{blue:02x}"

def animate_thinking_points(step):
    """
    step = 0:
    step = 1: .
    step = 2: ..
    step = 3: ...
    """
    global animation_job
    points = EXTRAS["thinking_points"]

    for i in range(3):
        level = (math.sin(step * 0.15 - i * 0.8) + 1) / 2
        canvas.itemconfig(points[i], fill=yellow_at(level))

    animation_job = root.after(40, animate_thinking_points, step+1)

def animate_angry():
    global animation_job, shake_sign, brow_shake
    shake_sign = -shake_sign
    shake_x = shake_sign * random.randint(ANGRY_SHAKE // 2, ANGRY_SHAKE)
    shake_y = random.randint(-2, 2)

    brow_shake = (shake_x, shake_y)
    place_brows()

    if not talking:  # en parlant, seuls les sourcils tremblent
        x1, y1, x2, y2 = EMOTIONS["angry"]["mouth_box"]
        canvas.coords(mouth_closed, x1 + shake_x // 2, y1, x2 + shake_x // 2, y2)
    animation_job = root.after(40, animate_angry)

def read_67_cadence():
    """Chiffre écrit par la danse, ou None si le fichier est absent, vide ou ancien."""
    try:
        if time.time() - CADENCE_67.stat().st_mtime > CADENCE_67_FRESH_S:
            return None
        text = CADENCE_67.read_text().strip()
    except OSError:
        return None
    return text if text in ("6", "7") else None

def animate_67():
    global animation_job, digit_phase, digit_x, digit_beat_start, digit_last_signal, brow_shake
    now = time.monotonic()

    # Un nouveau battement vient du fichier de cadence, ou de la cadence propre du visage.
    phase = read_67_cadence()
    if phase is not None:
        digit_last_signal = now
        if phase != digit_phase:
            digit_phase, digit_beat_start = phase, now
    elif (now - digit_last_signal > CADENCE_67_FRESH_S
          and now - digit_beat_start > CADENCE_67_FREE_MS / 1000):
        digit_phase, digit_beat_start = ("7" if digit_phase == "6" else "6"), now

    bounce = math.exp(-6 * (now - digit_beat_start))  # 1 au battement, puis retombe
    active = 0 if digit_phase == "6" else 1
    digit_x += ((-1.0 if active == 0 else 1.0) - digit_x) * 0.25  # balancement amorti
    blinking = now < digit_blink_until

    for k, (digit, cx) in enumerate(zip(EXTRAS["digits"], (200, 600), strict=True)):
        scale = 1.0 + 0.3 * bounce if k == active else 0.85
        canvas.coords(digit, cx + digit_x * DIGIT_SWAY, 195 - (14 * bounce if k == active else 0))
        canvas.itemconfig(digit, state="hidden" if blinking else "normal",
                          fill=DIGIT_YELLOW if k == active else DIGIT_PALE,
                          font=("Helvetica", -int(DIGIT_SIZE * scale), "bold"))

    # Les sourcils suivent le balancement, comme ils suivent les secousses de la colère.
    brow_shake = (digit_x * DIGIT_SWAY, -6 * bounce)
    place_brows()

    animation_job = root.after(30, animate_67)

def animate_talking():
    global talk_job, brow_lift
    animate_talking_mouth()
    brow_lift = -(talk_height - 6) / 74 * TALK_BROW_LIFT
    place_brows()
    talk_job = root.after(50, animate_talking)

def animate_talking_mouth():
    global talk_width, talk_height, talk_target

    if voix_bouche is not None:  # la voix donne le volume : la forme suit le son
        if voix_bouche < 0.1:
            names = ("close",)
        elif voix_bouche < 0.45:
            names = ("e", "i")
        else:
            names = ("a", "o")
        if talk_target not in [MOUTH_SHAPES[name] for name in names]:
            talk_target = MOUTH_SHAPES[random.choice(names)]
    elif abs(talk_width - talk_target["width"]) < 6 and abs(talk_height - talk_target["height"]) < 6:
        name = random.choices(
            list(MOUTH_SHAPES),
            weights=[shape["weight"] for shape in MOUTH_SHAPES.values()]
        )[0]
        talk_target = MOUTH_SHAPES[name]

    talk_width  += (talk_target["width"]  - talk_width)  * 0.5
    talk_height += (talk_target["height"] - talk_height) * 0.5

    cx, cy = TALK_CENTER

    if talk_height < 14: # bouche fermée
        canvas.coords(mouth_closed, cx - talk_width / 2, cy - talk_height / 2,
                             cx + talk_width / 2, cy + talk_height / 2)
        canvas.itemconfig(mouth_closed, start=0, extent=-180, style="arc",
                          fill="", outline="white", width=4, state="normal")
        canvas.itemconfig(mouth_open, state="hidden")
    else: # bouche ouverte
        canvas.coords(mouth_open, cx - talk_width / 2, cy - talk_height / 2,
                                cx + talk_width / 2, cy + talk_height / 2)
        canvas.itemconfig(mouth_open, fill="#ffffff", outline="", state="normal")
        canvas.itemconfig(mouth_closed, state="hidden")

def read_voice():
    """État écrit par la voix, ou None si le fichier est absent, illisible ou ancien."""
    try:
        if time.time() - VOIX.stat().st_mtime > VOIX_FRESH_S:
            return None
        state = json.loads(VOIX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None

def follow_voice():
    """Applique l'émotion et la parole voulues par la voix, seulement quand elles changent."""
    global voix_emotion, voix_parle, voix_bouche
    state = read_voice()

    if state is None:  # pas de voix : le visage reprend sa vie, bouche fermée
        if voix_parle:
            set_talking(False)
        voix_emotion, voix_parle, voix_bouche = None, False, None
    else:
        emotion = state.get("emotion")
        # set_emotion seulement au changement : sinon les animations repartiraient à zéro.
        if emotion in EMOTIONS and emotion != voix_emotion:
            set_emotion(emotion)
            voix_emotion = emotion

        parle = bool(state.get("parle"))
        try:
            voix_bouche = min(1.0, max(0.0, float(state.get("bouche", 0.0))))
        except (TypeError, ValueError):
            voix_bouche = None
        if parle != voix_parle:
            set_talking(parle)
            voix_parle = parle

    root.after(VOIX_PERIOD_MS, follow_voice)

def place_pupils(dx, dy):
    canvas.coords(left_pupil, 180, 180, 220, 220)  # remet les pupilles au centre
    canvas.coords(right_pupil, 580, 180, 620, 220)
    canvas.move(left_pupil, dx, dy)  # décale les pupilles
    canvas.move(right_pupil, dx, dy)

def blink():
    global digit_blink_until
    if not EMOTIONS[current_emotion].get("round_eyes", True):
        digit_blink_until = time.monotonic() + 0.15  # les chiffres clignent à la place
        return
    canvas.itemconfig(left_eye, fill="black")
    canvas.itemconfig(right_eye, fill="black")
    root.after(200, open_eyes)

def open_eyes():
    fiche = EMOTIONS[current_emotion]
    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

def move_eyes():
    if not EMOTIONS[current_emotion].get("round_eyes", True):
        root.after(random.randint(800, 2500), move_eyes)  # pas de pupilles en 67
        return
    dx, dy = EMOTIONS[current_emotion]["pupil_offset"]
    ecart_x = random.randint(-10, 10)
    ecart_y = random.randint(-10, 10)

    place_pupils(dx + ecart_x, dy + ecart_y)

    root.after(random.randint(800, 2500), move_eyes)

def random_blink():
    blink()
    root.after(random.randint(2000, 5000), random_blink)

def react(event):
    set_emotion("happy")
    print("Touched!")

    # son (si tu as branché un speaker)
    import os
    os.system("aplay /usr/share/sounds/alsa/Front_Left.wav &")

canvas.bind("<Button-1>", react)
root.bind("<space>", next_emotion)
root.bind("<Key-t>", toggle_talking)  # t : parler / se taire, en gardant l'émotion
root.bind("<Escape>", lambda event: root.destroy())

# Lancer animations
move_eyes()
random_blink()
set_emotion("happy")
follow_voice()
root.mainloop()
