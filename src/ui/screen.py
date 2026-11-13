import random
import tkinter as tk

root = tk.Tk()
root.attributes("-fullscreen", True)
root.configure(bg="black")

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Pour les sourcils (il n'apparaissent que pour certaines émotions)
left_brow = canvas.create_line(0, 0, 0, 0, fill="white", width=6, state="hidden")
right_brow = canvas.create_line(0, 0, 0, 0, fill="white", width=6, state="hidden")

# pour les yeux
left_eye = canvas.create_oval(150, 150, 250, 250, fill="white")
right_eye = canvas.create_oval(550, 150, 650, 250, fill="white")

# pour les pupille
left_pupil = canvas.create_oval(180, 180, 220, 220, fill="black")
right_pupil = canvas.create_oval(580, 180, 620, 220, fill="black")

# pour la bouche
mouth = canvas.create_arc(300, 280, 500, 400, start=0, extent=-180, style="arc", width=4, outline="white")

EMOTIONS = {
    "content" : {
        "mouth_box": (300, 280, 500, 400), #(x1, y1, x2, y2) boîte dans laquelle la bouche se trouve au centre
        "mouth_start": 0,
        "mouth_extent": -180,
        "eye_color": "white",
        "brows" : None,
        "pupil_offset" : (0, -8), # décalage de 8px vers le haut des pupilles
        "extras": ["joues"],
    },
    "triste" : {
        "mouth_box": (320, 330, 480, 410),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "#bcd4ff",
        "pupil_offset" : (0, 15),
        "brows" : ((150, 140, 250, 118), (550, 118, 650, 140)),
        "extras": ["larme"],
    },
    "reflechit" : {
        "mouth_box": (350, 339, 440, 341),
        "mouth_start": 0,
        "mouth_extent": 180,
        "eye_color": "white",
        "pupil_offset" : (15, -15),
        "brows" : ((150, 130, 250, 130), (550, 118, 650, 100)),
        "extras": ["points"],
    },
}

current_emotion = "content"
emotion_cycle = list(EMOTIONS)

def set_emotion(emotion):
    global current_emotion
    current_emotion = emotion

    fiche = EMOTIONS[emotion]

    canvas.coords(mouth, *fiche["mouth_box"])
    canvas.itemconfig(mouth, start=fiche["mouth_start"], extent=fiche["mouth_extent"])

    canvas.itemconfig(left_eye, fill=fiche["eye_color"])
    canvas.itemconfig(right_eye, fill=fiche["eye_color"])

    dx, dy = fiche["pupil_offset"]
    canvas.coords(left_pupil, 180, 180, 220, 220) #remet les pupilles au centre
    canvas.coords(right_pupil, 580, 180, 620, 220)
    canvas.move(left_pupil, dx, dy) # décale les pupilles
    canvas.move(right_pupil, dx, dy)

    if fiche["brows"] is None:
        canvas.itemconfig(left_brow, state="hidden")
        canvas.itemconfig(right_brow, state="hidden")
    else:
        canvas.coords(left_brow, *fiche["brows"][0])
        canvas.coords(right_brow, *fiche["brows"][1])

        canvas.itemconfig(left_brow, state="normal")
        canvas.itemconfig(right_brow, state="normal")

def next_emotion(event):
    next_emotion_index = (emotion_cycle.index(current_emotion) + 1)%len(emotion_cycle)
    set_emotion(emotion_cycle[next_emotion_index])

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
