import tkinter as tk
import random

root = tk.Tk()
root.attributes("-fullscreen", True)
root.configure(bg="black")

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

# pour les yeux
left_eye = canvas.create_oval(150, 150, 250, 250, fill="white")
right_eye = canvas.create_oval(550, 150, 650, 250, fill="white")

# pour les pupille
left_pupil = canvas.create_oval(180, 180, 220, 220, fill="black")
right_pupil = canvas.create_oval(580, 180, 620, 220, fill="black")

# pour la bouche
mouth = canvas.create_arc(300, 280, 500, 400, start=0, extent=-180, style="arc", width=4, outline="white")

def blink():
    canvas.itemconfig(left_eye, fill="black")
    canvas.itemconfig(right_eye, fill="black")
    root.after(200, open_eyes)

def open_eyes():
    canvas.itemconfig(left_eye, fill="white")
    canvas.itemconfig(right_eye, fill="white")

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
    canvas.itemconfig(mouth, extent=-180)  # sourire
    print("Touched!")

    # son (si tu as branché un speaker)
    import os
    os.system("aplay /usr/share/sounds/alsa/Front_Left.wav &")

canvas.bind("<Button-1>", react)


# Lancer animations
move_eyes()
random_blink()
change_mouth()
root.mainloop()