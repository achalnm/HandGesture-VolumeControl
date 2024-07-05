import cv2
import numpy as np
import subprocess
import tensorflow as tf
import keyboard
import time


def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)

# Function to detect thumbs up or thumbs down
def detect_thumb(frame, model):
    preprocessed_image = preprocess_image(frame)
    prediction = model.predict(preprocessed_image)
    if prediction[0][0] > 0.5:
        return "Thumbs up"
    else:xk
    current_volume = 50  
    thumb_detected = False  
    cooldown_time = 0.5
    last_thumb_time = time.time()  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture = detect_thumb(frame, model)
        print("Detected Gesture:", gesture)

        # Update thumb detected flag
        thumb_detected = (gesture == "Thumbs up" or gesture == "Thumbs down")

        if time.time() - last_thumb_time >= cooldown_time:
            if thumb_detected:
                if gesture == "Thumbs up":
                    keyboard.press_and_release('volume_up') 
                elif gesture == "Thumbs down":
                    keyboard.press_and_release('volume_down') 
                last_thumb_time = time.time() 

        cv2.imshow('Volume Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()