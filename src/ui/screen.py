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
mouth = canvas.create_arc(300, 280, 500, 400, start=0, extent=-180, style="arc", width=4, outline="white")

EXTRAS = {
    "cheeks": [left_cheek, right_cheek],
    "tear": [tear],
    "thinking_points": [thinking_point1, thinking_point2, thinking_point3],
}

EMOTIONS = {
    "content": {
        # (x1, y1, x2, y2) : boîte dans laquelle la bouche est dessinée
        "mouth_box": (300, 280, 500, 400),
        "mouth_start": 0,
        "mouth_extent": -180,
        "eye_color": "white",
        "brows": None,
        "pupil_offset": (0, -8),  # décalage de 8px vers le haut des pupilles
        "extras": ["cheeks"],
    },
    "triste": {
        "mouth_box": (320, 330, 480, 410),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "#bcd4ff",
        "pupil_offset": (0, 15),
        "brows": ((150, 140, 250, 118), (550, 118, 650, 140)),
        "extras": ["tear"],
    },
    "reflechit": {
        "mouth_box": (350, 339, 440, 341),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "white",
        "pupil_offset": (15, -15),
        "brows": ((150, 130, 250, 130), (550, 118, 650, 100)),
        "extras": ["thinking_points"],
    },
}

current_emotion = "content"
emotion_cycle = list(EMOTIONS)

animation_job = None

def set_emotion(emotion):
    global current_emotion
    global animation_job

    current_emotion = emotion

    fiche = EMOTIONS[emotion]

    canvas.coords(mouth, *fiche["mouth_box"])
    canvas.itemconfig(mouth, start=fiche["mouth_start"], extent=fiche["mouth_extent"])

    for extra in EXTRAS:
        for item in EXTRAS[extra]:
            canvas.itemconfig(item, state="hidden")
    for extra in fiche["extras"]:
        for item in EXTRAS[extra]:
            canvas.itemconfig(item, state="normal")

    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

    dx, dy = fiche["pupil_offset"]
    canvas.coords(left_pupil, 180, 180, 220, 220)  # remet les pupilles au centre
    canvas.coords(right_pupil, 580, 180, 620, 220)
    canvas.move(left_pupil, dx, dy)  # décale les pupilles
    canvas.move(right_pupil, dx, dy)

    if fiche["brows"] is None:
        canvas.itemconfig(left_brow, state="hidden")
        canvas.itemconfig(right_brow, state="hidden")
    else:
        canvas.coords(left_brow, *fiche["brows"][0])
        canvas.coords(right_brow, *fiche["brows"][1])

        canvas.itemconfig(left_brow, state="normal")
        canvas.itemconfig(right_brow, state="normal")

    # ANIMATIONS
    if animation_job is not None:
        root.after_cancel(animation_job)
        animation_job = None
    canvas.coords(tear, 175, 255, 195, 285)
    if emotion == "triste":
        animate_tear()
    elif emotion == "reflechit":
        animate_thinking_points(0)

def next_emotion(event):
    next_emotion_index = (emotion_cycle.index(current_emotion) + 1) % len(emotion_cycle)
    set_emotion(emotion_cycle[next_emotion_index])

def animate_tear():
    global animation_job
    canvas.move(tear, 0, 2)

    if canvas.coords(tear)[1] > 370:
        canvas.coords(tear, 175, 255, 195, 285)

    animation_job = root.after(50, animate_tear)

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
        if i < step:
            canvas.itemconfig(points[i], state="normal")
        else:
            canvas.itemconfig(points[i], state="hidden")

    step = (step + 1) % 4

    animation_job = root.after(400, animate_thinking_points, step)

def blink():
    canvas.itemconfig(left_eye, fill="black")
    canvas.itemconfig(right_eye, fill="black")
    root.after(200, open_eyes)

def open_eyes():
    fiche = EMOTIONS[current_emotion]
    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

def move_eyes():
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)

    for pupil in [left_pupil, right_pupil]:
        x1, y1, x2, y2 = canvas.coords(pupil)

        if 160 < x1 + dx < 240:
            canvas.move(pupil, dx, 0)
        if 160 < y1 + dy < 240:
            canvas.move(pupil, 0, dy)

    root.after(1000, move_eyes)

def change_mouth():
    mood = random.choice(["smile", "flat", "surprise"])

    if mood == "smile":
        canvas.itemconfig(mouth, extent=-180)
    elif mood == "flat":
        canvas.itemconfig(mouth, extent=0)
    else:
        canvas.itemconfig(mouth, extent=180)

    root.after(2000, change_mouth)

def random_blink():
    blink()
    root.after(random.randint(2000, 5000), random_blink)

def react(event):
    set_emotion("content")
    print("Touched!")

    # son (si tu as branché un speaker)
    import os
    os.system("aplay /usr/share/sounds/alsa/Front_Left.wav &")

canvas.bind("<Button-1>", react)
root.bind("<space>", next_emotion)

# Lancer animations
move_eyes()
random_blink()
set_emotion("content")
root.mainloop()
