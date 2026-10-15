
import time
import os
import csv
from collections import deque, Counter

import cv2
import numpy as np
from fer.fer import FER
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================
# 1. CUSTOM TKINTER
# =====================================================

try:
    import customtkinter as ctk
    USING_CTK = True
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    USING_CTK = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(SCRIPT_DIR, "emojis")

CSV_FILE = os.path.join(SCRIPT_DIR, "emotion_attention_log.csv")
DISTRACTION_CSV = os.path.join(SCRIPT_DIR, "distraction_log.csv")

os.makedirs(EMOJI_DIR, exist_ok=True)

# =====================================================
# 2. IMAGE FUNCTIONS
# =====================================================

def load_rgba_image(path):
    if not os.path.exists(path):
        print(f"[WARN] Image not found: {path}")
        return None

    try:
        with Image.open(path) as image:
            rgba = np.array(image.convert("RGBA"))

        # RGBA -> BGRA
        return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
    except Exception as e:
        print(f"[WARN] Image error: {e}")
        return None


def overlay_png(background, overlay, x, y, size=64):
    if overlay is None:
        return background

    if overlay.ndim != 3 or overlay.shape[2] != 4:
        return background

    overlay = cv2.resize(
        overlay, (size, size),
        interpolation=cv2.INTER_AREA
    )

    H, W = background.shape[:2]

    x, y = int(x), int(y)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + size)
    y2 = min(H, y + size)

    if x1 >= x2 or y1 >= y2:
        return background

    crop = overlay[
        y1 - y:y2 - y,
        x1 - x:x2 - x
    ]

    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0

    background[y1:y2, x1:x2] = (
        alpha * crop[:, :, :3] +
        (1 - alpha) * background[y1:y2, x1:x2]
    ).astype(np.uint8)

    return background


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


# =====================================================
# 3. AI MODELS
# =====================================================

detector = FER(mtcnn=False)

smooth_buffer = deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# =====================================================
# 4. EMOJIS
# =====================================================

emoji_files = {
    "angry": "angry.png",
    "disgust": "disgust.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "surprise": "surprise.png",
    "neutral": "neutral.png",
    "tired": "tired.png",
    "sleepy": "sleepy.png"
}

emoji_images = {
    name: load_rgba_image(os.path.join(EMOJI_DIR, filename))
    for name, filename in emoji_files.items()
}

# Phone icon is NOT an emotion
PHONE_ICON = load_rgba_image(
    os.path.join(EMOJI_DIR, "phone.png")
)

# =====================================================
# 5. GUI SETUP
# =====================================================

if USING_CTK:
    AppRoot = ctk.CTk
    FrameClass = ctk.CTkFrame
    ButtonClass = ctk.CTkButton
    LabelClass = ctk.CTkLabel
else:
    AppRoot = tk.Tk
    FrameClass = tk.Frame
    ButtonClass = tk.Button
    LabelClass = tk.Label


root = AppRoot()
root.title("Emotion & Attention Detector")
root.geometry("1100x780")
root.minsize(950, 650)

if not USING_CTK:
    root.configure(bg="#242424")

main_frame = FrameClass(root)
main_frame.pack(
    fill="both", expand=True,
    padx=12, pady=12
)

# LEFT SIDE
left = FrameClass(main_frame)
left.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)

# Camera
video_panel = tk.Label(
    left,
    text="Camera is OFF",
    font=("Arial", 18),
    bg="#171717",
    fg="white"
)

video_panel.pack(
    fill="both",
    expand=True,
    padx=8,
    pady=8
)

# Graph
fig, ax = plt.subplots(figsize=(6, 2.6))
fig.patch.set_facecolor("#242424")
ax.set_facecolor("#242424")

canvas = FigureCanvasTkAgg(fig, master=left)
canvas.get_tk_widget().pack(
    fill="x",
    padx=8,
    pady=8
)

# RIGHT SIDE
right = FrameClass(main_frame, width=260)
right.pack(side="right", fill="y")
right.pack_propagate(False)

LabelClass(
    right,
    text="Emojis"
).pack(pady=(10, 5))

emoji_preview_frames = FrameClass(right)
emoji_preview_frames.pack(
    padx=8,
    pady=(0, 8),
    fill="x"
)

emoji_tk = {}

for name, img in emoji_images.items():
    if img is None:
        emoji_tk[name] = None
        continue

    rgba = img[:, :, [2, 1, 0, 3]]
    pil = Image.fromarray(rgba)
    pil.thumbnail((42, 42), RESAMPLE)

    emoji_tk[name] = ImageTk.PhotoImage(pil)

for name in emoji_files:
    row = tk.Frame(
        emoji_preview_frames,
        bg="#2b2b2b"
    )
    row.pack(fill="x", pady=2)

    tk.Label(
        row,
        text=name.capitalize(),
        bg="#2b2b2b",
        fg="white",
        anchor="w",
        width=12
    ).pack(side="left", padx=5)

    icon_label = tk.Label(
        row,
        text="",
        bg="#2b2b2b",
        fg="white"
    )

    if emoji_tk[name] is not None:
        icon_label.configure(image=emoji_tk[name])
        icon_label.image = emoji_tk[name]
    else:
        icon_label.configure(text="-")

    icon_label.pack(side="right", padx=5)


# =====================================================
# 6. STATS
# =====================================================

emotion_times = {
    emo: 0.0 for emo in emoji_files
}

attention_percentage = 100.0

# Existing attention behaviour
attention_decay = 0.03
attention_recover = 0.01

running = False
cap = None

start_time = None
prev_time = None

last_emotion = "neutral"

# Distraction
distraction_events = []
distraction_state = "focused"
distraction_state_start = None

# Avoid duplicate CSV writes
saved_event_count = 0

# Tkinter callback management
frame_job = None

# =====================================================
# 7. CSV FILES
# =====================================================

EMOTION_FIELDS = [
    "timestamp", "emotion", "attention"
]

DISTRACTION_FIELDS = [
    "start_time", "end_time", "type", "note"
]


def ensure_csv(path, fields):
    if not os.path.exists(path):
        with open(
            path, "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fields
            )
            writer.writeheader()


ensure_csv(CSV_FILE, EMOTION_FIELDS)
ensure_csv(DISTRACTION_CSV, DISTRACTION_FIELDS)


def save_emotion(emotion):
    with open(
        CSV_FILE, "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=EMOTION_FIELDS
        )

        writer.writerow({
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "emotion": emotion,
            "attention": f"{attention_percentage:.2f}%"
        })


def save_distraction_events():
    global saved_event_count

    unsaved = distraction_events[saved_event_count:]

    if not unsaved:
        return

    with open(
        DISTRACTION_CSV, "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=DISTRACTION_FIELDS
        )
        writer.writerows(unsaved)

    saved_event_count = len(distraction_events)


# =====================================================
# 8. DISTRACTION FUNCTIONS
# =====================================================

def distraction_note(kind):
    notes = {
        "away": "Üz kamerada aşkarlanmayıb",
        "looking_side": "Üz kadrın yan hissəsindədir",
        "looking_down": "Üz kadrın aşağı hissəsindədir"
    }

    return notes.get(kind, "Normal")


def time_string(timestamp):
    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp)
    )


def close_distraction(end_timestamp):
    global distraction_state
    global distraction_state_start

    if (
        distraction_state == "focused" or
        distraction_state_start is None
    ):
        return

    distraction_events.append({
        "start_time": time_string(distraction_state_start),
        "end_time": time_string(end_timestamp),
        "type": distraction_state,
        "note": distraction_note(distraction_state)
    })

    distraction_state = "focused"
    distraction_state_start = None


def track_distraction(current_state, now):
    global distraction_state
    global distraction_state_start

    if current_state == distraction_state:
        return

    close_distraction(now)

    if current_state != "focused":
        distraction_state = current_state
        distraction_state_start = now


# =====================================================
# 9. FRAME UPDATE
# =====================================================

def update_frame():
    global prev_time
    global attention_percentage
    global frame_job
    global last_emotion

    frame_job = None

    if not running or cap is None:
        return

    ret, frame = cap.read()

    if not ret or frame is None:
        frame_job = root.after(100, update_frame)
        return

    if frame.ndim == 2:
        frame = cv2.cvtColor(
            frame, cv2.COLOR_GRAY2BGR
        )
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGRA2BGR
        )

    now = time.time()
    dt = max(0, now - prev_time)
    prev_time = now

    H, W = frame.shape[:2]

    # ------------------------------------------
    # ATTENTION + DISTRACTION
    # ------------------------------------------

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    status = "not_attentive"
    current_distraction = "focused"

    face_box = None

    if len(faces) == 0:
        current_distraction = "away"
    else:
        # Largest detected face
        fx, fy, fw, fh = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        face_box = (fx, fy, fw, fh)

        roi_gray = gray[
            fy:fy + fh,
            fx:fx + fw
        ]

        eyes = eye_cascade.detectMultiScale(
            roi_gray
        )

        if len(eyes) >= 1:
            status = "attentive"

            for ex, ey, ew, eh in eyes:
                cv2.rectangle(
                    frame,
                    (fx + ex, fy + ey),
                    (fx + ex + ew, fy + ey + eh),
                    (0, 255, 0),
                    2
                )

        center_x = fx + fw / 2
        center_y = fy + fh / 2

        dx = abs(center_x - W / 2) / W
        dy = center_y / H

        if dy > 0.70:
            current_distraction = "looking_down"
        elif dx > 0.25:
            current_distraction = "looking_side"
        else:
            current_distraction = "focused"

    # ------------------------------------------
    # ATTENTION SCORE
    # ------------------------------------------

    if status == "attentive":
        attention_percentage = min(
            100.0,
            attention_percentage + attention_recover
        )
    else:
        attention_percentage = max(
            0.0,
            attention_percentage - attention_decay
        )

    # ------------------------------------------
    # DISTRACTION TRACKING
    # ------------------------------------------

    track_distraction(current_distraction, now)

    # ------------------------------------------
    # EMOTION DETECTION
    # ------------------------------------------

    label_emotion = "neutral"
    box = None

    try:
        results = detector.detect_emotions(frame)

        if results:
            emotions = results[0]["emotions"]

            smooth_buffer.append(emotions)

            avg = {
                emotion: sum(
                    item.get(emotion, 0)
                    for item in smooth_buffer
                ) / len(smooth_buffer)
                for emotion in emotions
            }

            label_emotion = max(
                avg, key=avg.get
            )

            box = results[0]["box"]

    except Exception as e:
        print("[WARN] Emotion detection:", e)

    # ------------------------------------------
    # TIRED / SLEEPY RULES
    # ------------------------------------------

    if (
        attention_percentage < 91 and
        label_emotion in ["neutral", "sad"]
    ):
        label_emotion = "tired"

    if (
        attention_percentage < 82 and
        label_emotion in ["neutral", "sad", "tired"]
    ):
        label_emotion = "sleepy"

    last_emotion = label_emotion

    if label_emotion in emotion_times and box is not None:
        emotion_times[label_emotion] += dt

    # ------------------------------------------
    # EMOTION BOX + EMOJI
    # ------------------------------------------

    if box is not None:
        x, y, w, h = [int(v) for v in box]

        x = max(0, x)
        y = max(0, y)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label_emotion,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        frame = overlay_png(
            frame,
            emoji_images.get(label_emotion),
            x + w + 10,
            y,
            size=64
        )

    # ------------------------------------------
    # PHONE WARNING
    # ------------------------------------------

    if current_distraction == "looking_down":

        if face_box is not None:
            px, py, pw, ph = face_box

            icon_x = px + pw + 15
            icon_y = py + int(ph * 0.3)

            if icon_x + 60 > W:
                icon_x = px - 70
        else:
            icon_x = 20
            icon_y = 65

        frame = overlay_png(
            frame,
            PHONE_ICON,
            icon_x,
            icon_y,
            size=60
        )

        cv2.rectangle(
            frame,
            (10, 8),
            (min(W - 10, 480), 55),
            (0, 0, 100),
            -1
        )

        cv2.putText(
            frame,
            "WARNING: Looking down (Phone?)",
            (18, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # ------------------------------------------
    # CSV LOGGING
    # ------------------------------------------

    save_emotion(label_emotion)

    # ------------------------------------------
    # CAMERA DISPLAY
    # ------------------------------------------

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)
    pil_image.thumbnail((800, 600), RESAMPLE)

    imgtk = ImageTk.PhotoImage(pil_image)

    video_panel.configure(
        image=imgtk,
        text=""
    )

    video_panel.imgtk = imgtk

    frame_job = root.after(30, update_frame)


# =====================================================
# 10. CAMERA CONTROLS
# =====================================================


def start_camera():
    global cap, running
    global start_time, prev_time
    global distraction_events
    global distraction_state
    global distraction_state_start
    global saved_event_count
    global attention_percentage

    if running:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera açıla bilmədi.")
        cap = None
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    running = True

    start_time = time.time()
    prev_time = start_time

    distraction_events = []
    distraction_state = "focused"
    distraction_state_start = None
    saved_event_count = 0

    attention_percentage = 100.0
    smooth_buffer.clear()

    for name in emotion_times:
        emotion_times[name] = 0.0

    print("[INFO] Camera started.")
    update_frame()


def stop_camera():
    global running, cap, frame_job

    if not running:
        return

    running = False

    if frame_job is not None:
        root.after_cancel(frame_job)
        frame_job = None

    close_distraction(time.time())

    if cap is not None:
        cap.release()
        cap = None

    save_distraction_events()

    video_panel.configure(
        image="",
        text="Camera is OFF"
    )
    video_panel.imgtk = None

    print("\n===== EMOTION SUMMARY =====")

    for emotion, seconds in emotion_times.items():
        print(
            f"{emotion.capitalize()}: "
            f"{seconds:.1f} sec"
        )

    print("\n===== ATTENTION SUMMARY =====")
    print(f"Final Attention: {attention_percentage:.2f}%")

    print("\n===== DISTRACTION EVENTS =====")

    if not distraction_events:
        print("No distractions detected.")
    else:
        for i, event in enumerate(
            distraction_events, 1
        ):
            print(
                f"{i}. {event['start_time']} -> "
                f"{event['end_time']} | "
                f"{event['type']}"
            )

    print("[INFO] Camera stopped.")


# =====================================================
# 11. GRAPHS
# =====================================================

def show_graph():
    ax.clear()

    emotions = list(emotion_times.keys())
    times = list(emotion_times.values())

    ax.bar(
        emotions,
        times,
        color="#4DA3FF"
    )

    ax.set_title(
        "Emotion Duration (Seconds)",
        color="white"
    )

    ax.tick_params(
        axis="x",
        labelrotation=35,
        colors="white"
    )

    ax.tick_params(
        axis="y",
        colors="white"
    )

    for spine in ax.spines.values():
        spine.set_color("#666666")

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()
    canvas.draw()


def show_pie_from_csv():
    if not os.path.exists(CSV_FILE):
        print("[WARN] CSV not found.")
        return

    with open(
        CSV_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    counts = Counter(
        row["emotion"]
        for row in rows
        if row.get("emotion")
    )

    if not counts:
        print("[INFO] No emotion data.")
        return

    pie_fig, pie_ax = plt.subplots(
        figsize=(7, 6)
    )

    pie_ax.pie(
        list(counts.values()),
        labels=list(counts.keys()),
        autopct="%1.1f%%",
        startangle=90
    )

    pie_ax.set_title("Emotion Distribution")
    pie_fig.tight_layout()
    plt.show(block=False)


# =====================================================
# 12. RESET STATS
# =====================================================

def reset_stats():
    global attention_percentage
    global distraction_events
    global distraction_state
    global distraction_state_start
    global saved_event_count
    global prev_time

    for key in emotion_times:
        emotion_times[key] = 0.0

    attention_percentage = 100.0

    smooth_buffer.clear()

    distraction_events = []
    distraction_state = "focused"
    distraction_state_start = None
    saved_event_count = 0

    if running:
        prev_time = time.time()

    ax.clear()
    canvas.draw()

    # Clear current-session logs
    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=EMOTION_FIELDS
        )
        writer.writeheader()

    with open(
        DISTRACTION_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=DISTRACTION_FIELDS
        )
        writer.writeheader()

    print("[INFO] Statistics reset.")


# =====================================================
# 13. SHOW ATTENTION
# =====================================================

def show_attention():
    win = tk.Toplevel(root)
    win.title("Attention Score")
    win.geometry("410x230")
    win.configure(bg="#242424")

    tk.Label(
        win,
        text="Attention Score",
        bg="#242424",
        fg="white",
        font=("Arial", 17, "bold")
    ).pack(pady=12)

    value_label = tk.Label(
        win,
        text="",
        bg="#242424",
        fg="white",
        font=("Arial", 22, "bold")
    )
    value_label.pack(pady=8)

    progress = ttk.Progressbar(
        win,
        orient="horizontal",
        length=320,
        mode="determinate",
        maximum=100
    )
    progress.pack(pady=12)

    def refresh():
        if not win.winfo_exists():
            return

        value_label.configure(
            text=f"{attention_percentage:.2f}%"
        )

        progress["value"] = attention_percentage

        win.after(250, refresh)

    refresh()


# =====================================================
# 14. DISTRACTION REPORT
# =====================================================

def show_distractions():

    temp_events = list(distraction_events)

    if (
        running and
        distraction_state != "focused" and
        distraction_state_start is not None
    ):
        temp_events.append({
            "start_time": time_string(
                distraction_state_start
            ),
            "end_time": time_string(time.time()) +
                        " (ongoing)",
            "type": distraction_state,
            "note": distraction_note(
                distraction_state
            )
        })

    win = tk.Toplevel(root)
    win.title("Distraction Report")
    win.geometry("700x450")
    win.configure(bg="#242424")

    tk.Label(
        win,
        text="Distraction Report",
        font=("Arial", 18, "bold"),
        bg="#242424",
        fg="white"
    ).pack(pady=12)

    columns = (
        "start",
        "end",
        "type",
        "note"
    )

    table_frame = tk.Frame(win)
    table_frame.pack(
        fill="both",
        expand=True,
        padx=12,
        pady=10
    )

    tree = ttk.Treeview(
        table_frame,
        columns=columns,
        show="headings"
    )

    tree.heading("start", text="Start Time")
    tree.heading("end", text="End Time")
    tree.heading("type", text="Type")
    tree.heading("note", text="Description")

    tree.column("start", width=135)
    tree.column("end", width=165)
    tree.column("type", width=110)
    tree.column("note", width=245)

    scrollbar = ttk.Scrollbar(
        table_frame,
        orient="vertical",
        command=tree.yview
    )

    tree.configure(
        yscrollcommand=scrollbar.set
    )

    tree.pack(
        side="left",
        fill="both",
        expand=True
    )

    scrollbar.pack(
        side="right",
        fill="y"
    )

    for event in temp_events:
        tree.insert(
            "",
            "end",
            values=(
                event["start_time"],
                event["end_time"],
                event["type"],
                event["note"]
            )
        )

    if not temp_events:
        tk.Label(
            win,
            text="No distractions recorded.",
            bg="#242424",
            fg="white"
        ).pack(pady=5)

    tk.Button(
        win,
        text="Close",
        command=win.destroy
    ).pack(pady=10)


# =====================================================
# 15. BUTTONS
# =====================================================

ButtonClass(
    right,
    text="▶ Start Camera",
    command=start_camera
).pack(pady=6)

ButtonClass(
    right,
    text="■ Stop Camera",
    command=stop_camera
).pack(pady=6)

ButtonClass(
    right,
    text="📊 Show Graph",
    command=show_graph
).pack(pady=6)

ButtonClass(
    right,
    text="🥧 Show Pie Chart",
    command=show_pie_from_csv
).pack(pady=6)

ButtonClass(
    right,
    text="🔄 Reset Stats",
    command=reset_stats
).pack(pady=6)

ButtonClass(
    right,
    text="📈 Show Attention",
    command=show_attention
).pack(pady=6)

ButtonClass(
    right,
    text="📋 Show Distractions",
    command=show_distractions
).pack(pady=6)


# =====================================================
# 16. CLOSE APPLICATION
# =====================================================

def on_close():
    if running:
        stop_camera()

    plt.close("all")
    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)

# =====================================================
# START APPLICATION
# =====================================================

root.mainloop()
