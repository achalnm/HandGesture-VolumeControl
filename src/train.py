import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_data

# --- Constants ---
IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 16           # smaller batch suits the ~80-sample training set
VAL_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 1e-4
PATIENCE = 5              # EarlyStopping patience
UNFREEZE_LAST_N = 30      # number of MobileNetV2 layers to fine-tune
SMALL_DATASET_THRESHOLD = 30   # warn when validation set is below this
DEFAULT_CSV = 'data/labels.csv'
DEFAULT_MODEL_OUTPUT = 'outputs/model.h5'
OUTPUTS_DIR = 'outputs'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(img_size=IMG_SIZE, unfreeze_last_n=UNFREEZE_LAST_N):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    # Freeze all base layers, then unfreeze the last N for fine-tuning
    base.trainable = True
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # training=False keeps BatchNorm layers in inference mode, which prevents our tiny
    # dataset from corrupting ImageNet batch statistics in the frozen layers.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)


def create_train_generator():
    """Aggressive augmentation for a ~80-sample training set."""
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        # Applied after geometric transforms; converts [0,255] -> [-1,1] for MobileNetV2
        preprocessing_function=mobilenet_preprocess,
    )


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------

def save_training_plots(history, outputs_dir):
    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    for metric, title, fname in [
        ('accuracy', 'Training Accuracy', 'training_accuracy.png'),
        ('loss',     'Training Loss',     'training_loss.png'),
    ]:
        fig, ax = plt.subplots()
        ax.plot(epochs_ran, history.history[metric],           label='Train')
        ax.plot(epochs_ran, history.history[f'val_{metric}'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend()
        fig.savefig(os.path.join(outputs_dir, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"Training plots saved to {outputs_dir}/")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, outputs_dir):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Thumbs Down', 'Thumbs Up']

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    fig.tight_layout()
    path = os.path.join(outputs_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix  -> {path}")


def _plot_roc_curve(y_true, y_prob, auc_score, outputs_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right')

    path = os.path.join(outputs_dir, 'roc_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC curve         -> {path}")


def evaluate_model(model, X_val_prep, y_val, outputs_dir):
    """Compute full metrics, save plots, write metrics.json, print summary."""
    n_val = len(y_val)

    # --- Dataset size context ---
    print(f"\n{'='*54}")
    print(f"  EVALUATION  (validation set: {n_val} samples)")
    if n_val < SMALL_DATASET_THRESHOLD:
        print(f"  WARNING: only {n_val} validation samples (threshold: {SMALL_DATASET_THRESHOLD}).")
        print(f"  Metrics shown below may not generalise. Collect more data")
        print(f"  before drawing conclusions about real-world performance.")
    print(f"{'='*54}")

    # --- Predictions ---
    y_prob = model.predict(X_val_prep, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Scalar metrics ---
    acc       = float(np.mean(y_pred == y_val))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall    = float(recall_score(y_val, y_pred, zero_division=0))
    f1        = float(f1_score(y_val, y_pred, zero_division=0))
    auc_score = float(roc_auc_score(y_val, y_prob))

    # --- Plots ---
    print("Saving evaluation outputs:")
    _plot_confusion_matrix(y_val, y_pred, outputs_dir)
    _plot_roc_curve(y_val, y_prob, auc_score, outputs_dir)

    # --- metrics.json ---
    small_val = n_val < SMALL_DATASET_THRESHOLD
    metrics = {
        "validation_samples": n_val,
        "small_validation_set": small_val,
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "roc_auc":   round(auc_score, 4),
        "note": (
            f"Validation set contains only {n_val} samples. "
            "This is a small-dataset proof-of-concept. Metrics should be interpreted "
            "with caution and are not representative of production-level performance."
            if small_val else
            f"Validation set: {n_val} samples."
        ),
    }

    metrics_path = os.path.join(outputs_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON      -> {metrics_path}")

    # --- Summary ---
    flag = "  [!] small dataset, interpret with caution" if small_val else ""
    print(f"\n{'='*54}")
    print(f"  RESULTS SUMMARY{flag}")
    print(f"{'='*54}")
    print(f"  Validation samples : {n_val}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1-score           : {f1:.4f}")
    print(f"  ROC AUC            : {auc_score:.4f}")
    print(f"{'='*54}\n")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 gesture recognition model')
    parser.add_argument('--csv',    type=str, default=DEFAULT_CSV,          help='Path to labeled CSV')
    parser.add_argument('--output', type=str, default=DEFAULT_MODEL_OUTPUT, help='Output .h5 path')
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load raw [0,255] images, mobilenet_preprocess is applied inside the generator
    print(f"Loading data from {args.csv} ...")
    images, labels = preprocess_data(args.csv, normalize=False)
    n_up   = int(labels.sum())
    n_down = int(len(labels) - n_up)
    print(f"Loaded {len(images)} images  ({n_up} thumbs_up, {n_down} thumbs_down).")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # Preprocess validation set once, no augmentation, just mobilenet scaling
    X_val_prep = mobilenet_preprocess(X_val.copy())

    # Class weights to handle the slight 50/47 imbalance
    raw_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), raw_weights)}
    print(f"Class weights: {class_weight}")

    # Build and compile
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    total     = sum(tf.size(v).numpy() for v in model.variables)
    print(f"Trainable params: {trainable:,} / {total:,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Augmentation generator, generates infinite augmented batches from X_train
    datagen = create_train_generator()
    datagen.fit(X_train)
    train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

    print(f"\nTraining up to {EPOCHS} epochs  (early stop patience={PATIENCE}, "
          f"steps/epoch={steps_per_epoch}) ...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val_prep, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Training history JSON ---
    history_path = os.path.join(OUTPUTS_DIR, 'training_history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")

    # --- Training plots ---
    save_training_plots(history, OUTPUTS_DIR)

    # --- Full evaluation ---
    evaluate_model(model, X_val_prep, y_val, OUTPUTS_DIR)

    # --- TFLite export ---
    tflite_path = os.path.splitext(args.output)[0] + '.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

    print(f"Done. Best model: {args.output}")


if __name__ == '__main__':
    main()
