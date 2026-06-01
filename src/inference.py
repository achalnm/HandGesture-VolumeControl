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

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_SECONDS = 0.8
VOLUME_KEY_STEPS = 5
BORDER_THICKNESS = 8
DEFAULT_MODEL_PATH = 'outputs/models/model.h5'

VK_VOLUME_UP    = 0xAF
VK_VOLUME_DOWN  = 0xAE
KEYEVENTF_KEYUP = 0x0002

GREEN = (30,  210,  30)
RED   = (30,   30, 210)
AMBER = (0,   190, 230)
WHITE = (230, 230, 230)
GREY  = (150, 150, 150)
CYAN  = (0,   200, 255)


def preprocess_frame(frame, img_size=IMG_SIZE):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size)).astype(np.float32)
    return np.expand_dims(mobilenet_preprocess(resized), axis=0)


def detect_gesture(frame, model):
    raw = float(model.predict(preprocess_frame(frame), verbose=0)[0][0])
    if raw > CONFIDENCE_THRESHOLD:
        return 'Thumbs Up', raw
    if raw < 1.0 - CONFIDENCE_THRESHOLD:
        return 'Thumbs Down', 1.0 - raw
    return 'Unknown', max(raw, 1.0 - raw)


def draw_overlay(frame, gesture, confidence, volume, in_cooldown):
    h, w = frame.shape[:2]
    acting = gesture != 'Unknown'

    color = GREEN if acting else RED
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, BORDER_THICKNESS)
    cv2.putText(frame, gesture, (16, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {confidence:.0%}", (16, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)

    if acting and in_cooldown:
        cv2.putText(frame, "cooldown...", (16, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 1, cv2.LINE_AA)

    bx1, bx2    = w - 42, w - 18
    by_top, by_bot = BORDER_THICKNESS + 16, h - BORDER_THICKNESS - 32
    filled      = int(volume * (by_bot - by_top))

    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), (55, 55, 55), -1)
    if filled > 0:
        cv2.rectangle(frame, (bx1, by_bot - filled), (bx2, by_bot), CYAN, -1)
    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), GREY, 1)
    cv2.putText(frame, "VOL", (bx1 + 1, by_top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{volume:.0%}", (bx1 - 2, by_bot + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q: quit", (16, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    return frame


def get_volume_interface():
    device = AudioUtilities.GetSpeakers()
    if hasattr(device, 'EndpointVolume'):
        return device.EndpointVolume
    interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def change_volume(up, steps=VOLUME_KEY_STEPS):
    key = VK_VOLUME_UP if up else VK_VOLUME_DOWN
    for _ in range(steps):
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)
        ctypes.windll.user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run src/train.py first.")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    try:
        volume_ctrl = get_volume_interface()
    except Exception as e:
        print(f"Volume interface error: {e}")
        cap.release()
        sys.exit(1)

    last_action = 0.0
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, confidence = detect_gesture(frame, model)
        vol = volume_ctrl.GetMasterVolumeLevelScalar()
        now = time.time()
        in_cooldown = (now - last_action) < COOLDOWN_SECONDS

        if gesture != 'Unknown' and not in_cooldown:
            change_volume(up=(gesture == 'Thumbs Up'))
            last_action = now
            vol = volume_ctrl.GetMasterVolumeLevelScalar()

        draw_overlay(frame, gesture, confidence, vol, in_cooldown)
        cv2.imshow('Gesture Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
