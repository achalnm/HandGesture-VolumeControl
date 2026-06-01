import argparse
import ctypes
import os
import sys
import time
from ctypes import POINTER, cast

import cv2
import numpy as np
import tensorflow as tf
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# --- Constants ---
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75   # act on a gesture only when confidence exceeds this
COOLDOWN_SECONDS = 0.8        # minimum gap between volume changes
VOLUME_KEY_STEPS = 5          # key presses per gesture (each step = ~2%, so 5 = ~10%)
BORDER_THICKNESS = 8
DEFAULT_MODEL_PATH = 'outputs/model.h5'

# Windows virtual key codes for volume
VK_VOLUME_UP   = 0xAF
VK_VOLUME_DOWN = 0xAE
KEYEVENTF_KEYUP = 0x0002

# BGR colours used in the overlay
COLOR_GREEN  = (30,  210,  30)   # confident gesture
COLOR_RED    = (30,   30, 210)   # uncertain / no gesture
COLOR_AMBER  = (0,   190, 230)   # cooldown active
COLOR_WHITE  = (230, 230, 230)
COLOR_GREY   = (150, 150, 150)
COLOR_VOL    = (0,   200, 255)   # volume bar fill


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame, img_size=IMG_SIZE):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # match training pipeline
    resized   = cv2.resize(frame_rgb, (img_size, img_size)).astype(np.float32)
    return np.expand_dims(mobilenet_preprocess(resized), axis=0)  # [0,255] -> [-1,1]


# ---------------------------------------------------------------------------
# Gesture detection
# ---------------------------------------------------------------------------

def detect_gesture(frame, model):
    """Return (label, confidence).  label is 'Thumbs Up', 'Thumbs Down', or 'Unknown'."""
    raw = float(model.predict(preprocess_frame(frame), verbose=0)[0][0])
    if raw > CONFIDENCE_THRESHOLD:
        return 'Thumbs Up',   raw
    if raw < 1.0 - CONFIDENCE_THRESHOLD:
        return 'Thumbs Down', 1.0 - raw
    return 'Unknown', max(raw, 1.0 - raw)


# ---------------------------------------------------------------------------
# Display overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, gesture, confidence, volume, in_cooldown):
    h, w = frame.shape[:2]
    acting = gesture != 'Unknown'

    # --- Border: green = confident gesture, red = uncertain ---
    border_color = COLOR_GREEN if acting else COLOR_RED
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, BORDER_THICKNESS)

    # --- Gesture label + confidence (top-left) ---
    label_color = COLOR_GREEN if acting else COLOR_RED
    cv2.putText(frame, gesture,
                (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence:.0%}",
                (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

    # Cooldown badge (shown when gesture is active but cooldown is not yet elapsed)
    if acting and in_cooldown:
        cv2.putText(frame, "cooldown...",
                    (16, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_AMBER, 1, cv2.LINE_AA)

    # --- Volume bar (right side, inside the border) ---
    bx1, bx2      = w - 42, w - 18
    by_top, by_bot = BORDER_THICKNESS + 16, h - BORDER_THICKNESS - 32
    bar_h          = by_bot - by_top
    filled_h       = int(volume * bar_h)

    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), (55, 55, 55), -1)     # track
    if filled_h > 0:
        cv2.rectangle(frame,
                      (bx1, by_bot - filled_h), (bx2, by_bot),
                      COLOR_VOL, -1)                                            # fill
    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), COLOR_GREY, 1)         # outline

    # Labels above and below the bar
    cv2.putText(frame, "VOL",
                (bx1 + 1, by_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{volume:.0%}",
                (bx1 - 2, by_bot + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # --- Quit hint (bottom-left) ---
    cv2.putText(frame, "Q: quit",
                (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREY, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Volume interface
# ---------------------------------------------------------------------------

def get_volume_interface():
    device = AudioUtilities.GetSpeakers()
    # Newer pycaw wraps IMMDevice in an AudioDevice class with EndpointVolume property.
    # Older pycaw returned the raw COM object which had .Activate() directly.
    if hasattr(device, 'EndpointVolume'):
        return device.EndpointVolume
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


# ---------------------------------------------------------------------------
# Volume control (keyboard events so the native Windows OSD appears)
# ---------------------------------------------------------------------------

def change_volume(up, steps=VOLUME_KEY_STEPS):
    key = VK_VOLUME_UP if up else VK_VOLUME_DOWN
    for _ in range(steps):
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)              # key down
        ctypes.windll.user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0) # key up


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Real-time gesture-based volume control')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to trained .h5 model file')
    args = parser.parse_args()

    # --- Model ---
    if not os.path.exists(args.model):
        print(f"Error: model not found at '{args.model}'.")
        print("Train one first:  python src/train.py")
        sys.exit(1)

    try:
        print(f"Loading model from {args.model} ...")
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Camera ---
    print("Opening webcam ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        print("Check that a camera is connected and not already in use by another application.")
        sys.exit(1)

    # --- Windows volume ---
    try:
        volume_ctrl = get_volume_interface()
    except Exception as e:
        print(f"Error initialising pycaw volume interface: {e}")
        cap.release()
        sys.exit(1)

    last_action_time = 0.0
    print(f"Running | confidence threshold: {CONFIDENCE_THRESHOLD:.0%}  |  "
          f"cooldown: {COOLDOWN_SECONDS}s  |  press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read a frame from the webcam.")
            break

        gesture, confidence = detect_gesture(frame, model)
        current_vol = volume_ctrl.GetMasterVolumeLevelScalar()

        now         = time.time()
        in_cooldown = (now - last_action_time) < COOLDOWN_SECONDS

        if gesture != 'Unknown' and not in_cooldown:
            change_volume(up=(gesture == 'Thumbs Up'))
            last_action_time = now
            current_vol = volume_ctrl.GetMasterVolumeLevelScalar()

        draw_overlay(frame, gesture, confidence, current_vol, in_cooldown)
        cv2.imshow('Gesture Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == '__main__':
    main()
