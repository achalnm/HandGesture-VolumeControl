import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split #pip install scikit-learn #also used only to split training and testing data
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume #pip install Open-cv pycaw
import time


def preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for index, row in data.iterrows():
        filename = row['Full Path']
        folder = row['Folder']
        try:
            image = cv2.imread(filename)
            if image is None:
                print(f"Warning: Image '{filename}' could not be loaded.")
                continue
            image = cv2.resize(image, (224, 224))
            images.append(image)
            if folder == 'thumbs_up':
                labels.append(1)
            elif folder == 'thumbs_down':
                labels.append(0)
        except Exception as e:
            print(f"Error processing image '{filename}': {str(e)}")
            continue

    return np.array(images), np.array(labels)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    return model


def train_model(images, labels):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    # Display accuracy
    _, accuracy = model.evaluate(X_val, y_val)
    print("Validation Accuracy:", accuracy)

    return model


def main():
    csv_file = r'C:\Users\achal\Downloads\VolumeControl-main\RawData.csv'  # Modify this line with your CSV file location
    images, labels = preprocess_data(csv_file)
    model = train_model(images, labels)

    test_image = images[0]
    prediction = model.predict(np.expand_dims(test_image, axis=0))

    print("Prediction: ", prediction)
    model.save(r'C:\Users\achal\Downloads\VolumeControl-main\full_model.weights.h5') #Modify this line to save your model config

if __name__ == "__main__":
    main()

