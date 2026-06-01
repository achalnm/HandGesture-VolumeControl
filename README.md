# Hand Gesture Volume Control

A real-time Windows volume controller driven by hand gestures. A MobileNetV2 model fine-tuned via transfer learning watches a live webcam feed, classifies each frame as thumbs-up, thumbs-down, or uncertain, and adjusts system volume through the Windows audio API whenever a high-confidence gesture is detected. A confidence threshold of 0.75 and an 0.8-second cooldown prevent false triggers, and the OpenCV display window shows a green/red border, confidence score, and a live volume bar so the system state is always visible.

This project was developed as part of the **Bachelor of Engineering in Computer Science (BE CSE)** curriculum, under the **Data Visualization and Model Training** course.

---

## Results

Model trained on 388 images (201 thumbs-up, 187 thumbs-down), evaluated on a held-out validation set of 78 images (80/20 stratified split). EarlyStopping halted training at epoch 10; best checkpoint was restored from epoch 5.

| Metric | Value |
| --- | --- |
| Accuracy | **0.9615** |
| Precision | **0.9512** |
| Recall | **0.9750** |
| F1-score | **0.9630** |
| ROC AUC | **0.9737** |

> **Note:** These metrics are reported on the validation split used during training. Performance on entirely unseen data or different lighting conditions may vary. See `outputs/metrics.json` for the full machine-readable report.

### Confusion Matrix

![Confusion Matrix](outputs/plots/confusion_matrix.png)

### ROC Curve (AUC = 0.97)

![ROC Curve](outputs/plots/roc_curve.png)

### Training Accuracy

![Training Accuracy](outputs/plots/training_accuracy.png)

---

## Dataset

The training data consists of 388 images across two classes:

| Class | Count | Source |
| --- | --- | --- |
| Thumbs Up | 201 | Personal webcam photos + Bing Image Search |
| Thumbs Down | 187 | Personal webcam photos + Bing Image Search |

Personal photos were captured using the built-in Flask data collection app (`app.py`). Additional images were downloaded using `src/download_data.py`, which queries Bing Image Search with multiple search terms to get variety in backgrounds, lighting, and hand shapes.

### Using your own dataset

You can retrain the model on your own images to improve accuracy for your specific hand shape, skin tone, and lighting conditions.

1. Place your thumbs-up images in `data/images/thumbs_up/`
2. Place your thumbs-down images in `data/images/thumbs_down/`
3. Regenerate the CSV and retrain:

```bash
python src/preprocess.py --output data/labels.csv
python src/train.py --csv data/labels.csv --output outputs/models/model.h5
```

More images generally means better accuracy. Aim for at least 100 images per class, with varied backgrounds and lighting. You can also point `src/download_data.py` at any search term to bulk-download images from Bing automatically.

---

## System Architecture

```text
Webcam frame (BGR)
    |
    v
cv2.cvtColor(BGR -> RGB)         # match MobileNetV2 ImageNet training colour space
    |
    v
cv2.resize(224 x 224)
    |
    v
mobilenet_v2.preprocess_input()  # scale [0, 255] to [-1, 1]
    |
    v
MobileNetV2 (ImageNet weights)   # feature extractor, last 30 layers unfrozen
    |
    v
GlobalAveragePooling2D
Dense(128, relu) -> Dropout(0.3)
Dense(1, sigmoid)                # output: probability of thumbs-up
    |
    +-- output > 0.75  ->  Thumbs Up   ->  volume up (~10%)
    +-- output < 0.25  ->  Thumbs Down ->  volume down (~10%)
    +-- 0.25 to 0.75   ->  Unknown     ->  no action
```

The confidence threshold (0.75) and cooldown (0.8 s) are constants at the top of `src/inference.py` and can be adjusted without retraining.

---

## Project Structure

```text
app.py                          # Flask web app for uploading and labelling training images
src/
    train.py                    # MobileNetV2 fine-tuning, evaluation, TFLite export
    inference.py                # Real-time webcam inference + Windows volume control
    preprocess.py               # Scan image folders, generate labelled CSV
    download_data.py            # Download training images from Bing
data/
    images/
        thumbs_up/              # Thumbs-up training images
        thumbs_down/            # Thumbs-down training images
    labels.csv                  # Generated image labels (gitignored)
outputs/
    models/
        model.h5                # Trained Keras model (ready to use)
        model.tflite            # TFLite export for edge deployment
    plots/
        confusion_matrix.png
        roc_curve.png
        training_accuracy.png
        training_loss.png
    metrics.json
    training_history.json
templates/                      # HTML templates for the Flask data collection app
requirements.txt
```

---

## Requirements

- **OS:** Windows (volume control via pycaw is Windows-only; training and data collection work on any OS)
- **Python:** 3.8 exactly. TensorFlow 2.13 does not support Python 3.9+ on the Windows binary, and Python 3.10+ breaks several dependencies.

Check your available Python versions with `py -0` (Windows Launcher).

---

## Installation

```bash
git clone https://github.com/achalnm/HandGesture-VolumeControl.git
cd HandGesture-VolumeControl

py -3.8 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

The trained model is already committed to the repo (`outputs/models/model.h5`), so you can skip straight to inference without retraining.

---

## Usage

### 1. Collect training data

Run the Flask data collection app:

```bash
python app.py
```

Open `http://localhost:5000`, select your image folders, upload them, and download the generated CSV. It saves to `data/` automatically.

Or download images from Bing directly:

```bash
python src/download_data.py --per_query 60
```

### 2. Train the model

```bash
python src/train.py --csv data/labels.csv --output outputs/models/model.h5
```

Training runs for up to 50 epochs with EarlyStopping. After training, the following are written automatically:

- `outputs/models/model.h5` and `outputs/models/model.tflite`
- `outputs/plots/` — confusion matrix, ROC curve, training curves
- `outputs/metrics.json` and `outputs/training_history.json`

### 3. Run real-time inference

```bash
python src/inference.py --model outputs/models/model.h5
```

The OpenCV window shows:

- **Green border**: gesture detected above the confidence threshold
- **Red border**: no confident gesture (uncertain or no hand visible)
- **Amber badge**: gesture detected but cooldown has not elapsed yet
- **Volume bar** (right side): current system master volume
- **Confidence %**: model confidence for the detected class

Press **Q** to quit.

---

## Configuration

| Constant | File | Default | Description |
| --- | --- | --- | --- |
| `CONFIDENCE_THRESHOLD` | `src/inference.py` | `0.75` | Minimum confidence to act on a gesture |
| `COOLDOWN_SECONDS` | `src/inference.py` | `0.8` | Minimum gap between volume changes |
| `VOLUME_KEY_STEPS` | `src/inference.py` | `5` | Key presses per gesture (each ~2%) |
| `IMG_SIZE` | all `src/` files | `224` | Input frame size |
| `EPOCHS` | `src/train.py` | `50` | Max training epochs |
| `LEARNING_RATE` | `src/train.py` | `1e-4` | Adam learning rate |
| `UNFREEZE_LAST_N` | `src/train.py` | `30` | MobileNetV2 layers unfrozen for fine-tuning |
| `PATIENCE` | `src/train.py` | `5` | EarlyStopping patience |

---

## Tech Stack

| Component | Library / Version |
| --- | --- |
| Language | Python 3.8 |
| Deep learning | TensorFlow 2.13 / Keras 2.13 |
| Base model | MobileNetV2 (ImageNet pretrained) |
| Computer vision | OpenCV 4.8 |
| Windows audio | pycaw + comtypes |
| Data handling | pandas, NumPy, scikit-learn |
| Data collection UI | Flask 3.0 |
| Evaluation plots | matplotlib |

---

## Notes

**Why MobileNetV2?** Transfer learning from a model pretrained on 1.2 million ImageNet images lets the network extract strong visual features even from a small dataset. A CNN trained from scratch on a few hundred images overfits immediately.

**Extending the model.** The two-class setup (thumbs-up / thumbs-down) is intentional for simplicity, but the architecture extends naturally. Adding a third gesture class requires placing images in a new folder, updating the label logic in `src/preprocess.py`, and changing the output layer in `src/train.py` from sigmoid to softmax.

**Windows only (for inference).** The `pycaw` library wraps the Windows Core Audio API and is not available on macOS or Linux. The training pipeline and data collection app run on any OS.
