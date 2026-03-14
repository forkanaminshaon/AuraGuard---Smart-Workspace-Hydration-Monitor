"""
╔══════════════════════════════════════════════════════════════════╗
║          AuraGuard — Smart Workspace & Hydration Monitor         ║
║          Built with YOLOv8 + OpenCV | Runs on Google Colab       ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (run these in a Colab cell BEFORE this script):
    !pip install ultralytics opencv-python-headless
    from google.colab.patches import cv2_imshow   # used below automatically

HOW TO RUN IN COLAB:
    - Webcam mode:  set SOURCE = 0
    - Video file:   set SOURCE = "/content/your_video.mp4"

THRESHOLD (kept low for demo):
    PHONE_DISTRACTION_THRESHOLD  = 5   seconds
    HYDRATION_REMINDER_THRESHOLD = 30  seconds
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import cv2
import time
import sys
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIGURATION  ← tweak these freely
# ─────────────────────────────────────────────
SOURCE = 0                          # 0 = webcam  |  "/content/video.mp4" for file

MODEL_PATH = "yolov8n.pt"           # nano = fast; swap to yolov8s.pt for better accuracy

# Demo-friendly thresholds (seconds)
PHONE_DISTRACTION_THRESHOLD  = 5    # alert after phone visible for N seconds
HYDRATION_REMINDER_THRESHOLD = 30   # alert after N seconds without seeing cup/bottle

# ─────────────────────────────────────────────
#  COCO CLASS IDs WE CARE ABOUT
#  Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# ─────────────────────────────────────────────
CLASS_IDS = {
    0:  "person",
    67: "cell phone",
    39: "bottle",
    41: "cup",
}

# Which IDs count as "hydration objects"
HYDRATION_IDS  = {39, 41}   # bottle, cup
PHONE_IDS      = {67}       # cell phone
PERSON_IDS     = {0}        # person

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (BGR for OpenCV)
# ─────────────────────────────────────────────
COLORS = {
    "ok":      (50,  205,  50),   # lime green
    "warning": (0,   165, 255),   # orange
    "danger":  (0,    0,  220),   # red
    "info":    (255, 200,   0),   # cyan-ish
    "box_phone":    (0,   0, 255),
    "box_hydration":(0, 255,   0),
    "box_person":   (255, 200,  0),
}

# ─────────────────────────────────────────────
#  COLAB DETECTION — use cv2_imshow instead of cv2.imshow
# ─────────────────────────────────────────────
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
    print("✅  Google Colab detected — using cv2_imshow for display.")
except ImportError:
    IN_COLAB = False
    print("💻  Local environment detected — using cv2.imshow.")


# ══════════════════════════════════════════════
#  HELPER: Draw rounded rectangle (panel background)
# ══════════════════════════════════════════════
def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=0.5):
    """Draw a semi-transparent filled rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ══════════════════════════════════════════════
#  HELPER: Draw HUD panel
# ══════════════════════════════════════════════
def draw_hud(frame, states):
    """
    Overlay the AuraGuard HUD on the frame.

    states dict keys:
        person_detected   bool
        phone_visible     bool
        phone_seconds     float   (how long phone has been visible)
        hydration_ok      bool
        hydration_seconds float   (seconds since last drink seen)
        fps               float
    """
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_small = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top banner ──────────────────────────────
    draw_rounded_rect(frame, (0, 0), (w, 38), (20, 20, 20), radius=0, alpha=0.6)
    cv2.putText(frame, "AuraGuard  |  Smart Workspace Monitor",
                (10, 26), font_small, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    fps_text = f"FPS: {states['fps']:.1f}"
    cv2.putText(frame, fps_text, (w - 100, 26),
                font_small, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Status panel (bottom left) ───────────────
    panel_x, panel_y = 10, h - 130
    draw_rounded_rect(frame, (panel_x, panel_y), (panel_x + 310, h - 10),
                      (20, 20, 20), radius=8, alpha=0.65)

    # — Person status
    p_color = COLORS["ok"] if states["person_detected"] else COLORS["warning"]
    p_text  = "👤 Person: DETECTED" if states["person_detected"] else "👤 Person: NOT FOUND"
    cv2.putText(frame, p_text, (panel_x + 10, panel_y + 28),
                font_small, 0.55, p_color, 1, cv2.LINE_AA)

    # — Phone / distraction status
    if states["phone_visible"]:
        secs = states["phone_seconds"]
        if secs >= PHONE_DISTRACTION_THRESHOLD:
            ph_color = COLORS["danger"]
            ph_text  = f"📱 Phone: DISTRACTED  {secs:.0f}s"
        else:
            ph_color = COLORS["warning"]
            ph_text  = f"📱 Phone: detected  {secs:.0f}s"
    else:
        ph_color = COLORS["ok"]
        ph_text  = "📱 Phone: Away — Focused!"
    cv2.putText(frame, ph_text, (panel_x + 10, panel_y + 58),
                font_small, 0.55, ph_color, 1, cv2.LINE_AA)

    # — Hydration status
    h_secs = states["hydration_seconds"]
    if not states["hydration_ok"]:
        hy_color = COLORS["danger"]
        hy_text  = f"💧 Hydration: DRINK NOW!  {h_secs:.0f}s ago"
    elif h_secs > HYDRATION_REMINDER_THRESHOLD * 0.6:
        hy_color = COLORS["warning"]
        hy_text  = f"💧 Hydration: Soon...  {h_secs:.0f}s ago"
    else:
        hy_color = COLORS["ok"]
        hy_text  = f"💧 Hydration: Good  {h_secs:.0f}s ago"
    cv2.putText(frame, hy_text, (panel_x + 10, panel_y + 88),
                font_small, 0.55, hy_color, 1, cv2.LINE_AA)

    # ── Big alert banners ───────────────────────
    alert_y = 60
    if states["phone_visible"] and states["phone_seconds"] >= PHONE_DISTRACTION_THRESHOLD:
        draw_rounded_rect(frame, (w//2 - 240, alert_y), (w//2 + 240, alert_y + 44),
                          (0, 0, 180), radius=8, alpha=0.8)
        cv2.putText(frame, "⚠  PUT DOWN YOUR PHONE!",
                    (w//2 - 195, alert_y + 30), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        alert_y += 55

    if not states["hydration_ok"]:
        draw_rounded_rect(frame, (w//2 - 220, alert_y), (w//2 + 220, alert_y + 44),
                          (10, 100, 180), radius=8, alpha=0.8)
        cv2.putText(frame, "💧  TIME TO DRINK WATER!",
                    (w//2 - 185, alert_y + 30), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  MAIN APPLICATION LOOP
# ══════════════════════════════════════════════
def run():
    print("\n🚀  Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print(f"✅  Model loaded: {MODEL_PATH}")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"❌  Cannot open video source: {SOURCE}")
        sys.exit(1)

    print(f"📷  Video source opened: {SOURCE}")
    print("    Press  Q  to quit.\n")

    # ── Temporal state variables ─────────────────
    phone_start_time  = None      # when phone first appeared this session
    last_drink_time   = time.time()  # assume user just drank at launch

    fps_counter       = 0
    fps_timer         = time.time()
    current_fps       = 0.0

    # ── For Colab: limit display frame rate to avoid kernel overload
    colab_display_interval = 0.15   # show a frame every ~150 ms in Colab
    last_display_time      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠  End of video stream.")
            break

        now = time.time()

        # ── FPS counter ──────────────────────────
        fps_counter += 1
        if now - fps_timer >= 1.0:
            current_fps = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # ── Run YOLO inference ───────────────────
        results = model(frame, verbose=False)[0]

        # ── Parse detections ─────────────────────
        phone_in_frame    = False
        hydration_in_frame = False
        person_in_frame   = False

        for box in results.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id not in CLASS_IDS or confidence < 0.35:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{CLASS_IDS[cls_id]}  {confidence:.0%}"

            # Choose colour & flag
            if cls_id in PHONE_IDS:
                phone_in_frame = True
                color = COLORS["box_phone"]
            elif cls_id in HYDRATION_IDS:
                hydration_in_frame = True
                color = COLORS["box_hydration"]
            elif cls_id in PERSON_IDS:
                person_in_frame = True
                color = COLORS["box_person"]
            else:
                color = (180, 180, 180)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_bg_y = max(y1 - 22, 0)
            cv2.rectangle(frame, (x1, label_bg_y), (x1 + len(label) * 9, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Update temporal logic ─────────────────

        # Phone tracker
        if phone_in_frame:
            if phone_start_time is None:
                phone_start_time = now         # phone just appeared
            phone_seconds = now - phone_start_time
        else:
            phone_start_time = None            # phone gone — reset
            phone_seconds    = 0.0

        # Hydration tracker
        if hydration_in_frame:
            last_drink_time = now              # reset timer whenever cup/bottle seen
        hydration_seconds = now - last_drink_time
        hydration_ok      = hydration_seconds < HYDRATION_REMINDER_THRESHOLD

        # ── Build state dict for HUD ─────────────
        states = {
            "person_detected":  person_in_frame,
            "phone_visible":    phone_in_frame,
            "phone_seconds":    phone_seconds,
            "hydration_ok":     hydration_ok,
            "hydration_seconds": hydration_seconds,
            "fps":              current_fps,
        }

        # ── Render HUD ───────────────────────────
        draw_hud(frame, states)

        # ── Display ──────────────────────────────
        if IN_COLAB:
            # Colab: show every ~150 ms to avoid flooding output
            if now - last_display_time >= colab_display_interval:
                cv2_imshow(frame)
                last_display_time = now
            # Allow interrupt via keyboard (Colab cell stop button)
        else:
            cv2.imshow("AuraGuard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("👋  Q pressed — shutting down.")
                break

    cap.release()
    if not IN_COLAB:
        cv2.destroyAllWindows()
    print("✅  AuraGuard stopped cleanly.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run()