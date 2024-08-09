# 📈 Volume Control with Hand Gestures

Welcome to the **Volume Control with Hand Gestures** project! This innovative project leverages machine learning and computer vision to control your PC's volume using simple hand gestures. With this solution, you can adjust the volume by simply giving a thumbs up or thumbs down.

---

## 🌟 Features

- **Gesture Recognition:** Detects thumbs up and thumbs down gestures to control volume.
- **Real-Time Processing:** Processes webcam feed in real-time to adjust volume.
- **Machine Learning Model:** Utilizes a deep learning model trained on labeled image data.
- **Seamless Integration:** Works directly with your PC's volume controls.

---

## 💻 Technologies Used

- **Python**: The core programming language for the project.
- **Flask**: For creating a simple web interface for file uploads.
- **TensorFlow**: Used to build and train the deep learning model.
- **OpenCV**: Handles real-time image processing.
- **scikit-learn**: For splitting data into training and testing sets.
- **PyCaw**: Manages audio volume control on Windows.

---

## 📂 Project Structure

- `app.py`: Flask application for file handling and processing.
- `index.html`: Web interface for uploading image folders.
- `save_as.html`: Interface for specifying CSV file name and location.
- `main.py`: Data preprocessing, model training, and saving.
- `volume.py`: Real-time gesture detection and volume control.

---

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/volume-control.git
   cd volume-control
