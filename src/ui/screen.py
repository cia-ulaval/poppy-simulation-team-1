import math
import random
import tkinter as tk

root = tk.Tk()
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

# pour la bouche
mouth_closed = canvas.create_arc(300, 280, 500, 400, start=0, extent=-180, style="arc", width=4, outline="white")
mouth_open = canvas.create_oval(0, 0, 0, 0, fill="white", outline="", state="hidden")

EXTRAS = {
    "cheeks": [left_cheek, right_cheek],
    "tear": [tear],
    "thinking_points": [thinking_point1, thinking_point2, thinking_point3],
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
}

ANGRY_SHAKE = 5

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

def set_emotion(emotion):
    global current_emotion
    global animation_job
    global brow_shake

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

def animate_talking():
    global talk_job, brow_lift
    animate_talking_mouth()
    brow_lift = -(talk_height - 6) / 74 * TALK_BROW_LIFT
    place_brows()
    talk_job = root.after(50, animate_talking)

def animate_talking_mouth():
    global talk_width, talk_height, talk_target

    if abs(talk_width - talk_target["width"]) < 6 and abs(talk_height - talk_target["height"]) < 6:
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

def place_pupils(dx, dy):
    canvas.coords(left_pupil, 180, 180, 220, 220)  # remet les pupilles au centre
    canvas.coords(right_pupil, 580, 180, 620, 220)
    canvas.move(left_pupil, dx, dy)  # décale les pupilles
    canvas.move(right_pupil, dx, dy)

def blink():
    canvas.itemconfig(left_eye, fill="black")
    canvas.itemconfig(right_eye, fill="black")
    root.after(200, open_eyes)

def open_eyes():
    fiche = EMOTIONS[current_emotion]
    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

def move_eyes():
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

# Lancer animations
move_eyes()
random_blink()
set_emotion("happy")
root.mainloop()
