import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_data

IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 16
VAL_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 1e-4
PATIENCE = 5
UNFREEZE_LAST_N = 30
DEFAULT_CSV = 'data/labels.csv'
DEFAULT_MODEL_OUTPUT = 'outputs/model.h5'
OUTPUTS_DIR = 'outputs'


def build_model(img_size=IMG_SIZE):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)


def get_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        preprocessing_function=mobilenet_preprocess,
    )


def plot_history(history, out_dir):
    for metric, fname in [('accuracy', 'training_accuracy.png'), ('loss', 'training_loss.png')]:
        fig, ax = plt.subplots()
        ax.plot(history.history[metric], label='Train')
        ax.plot(history.history[f'val_{metric}'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)


def evaluate(model, X_val, y_val, out_dir):
    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = float(np.mean(y_pred == y_val))
    prec = float(precision_score(y_val, y_pred, zero_division=0))
    rec  = float(recall_score(y_val, y_pred, zero_division=0))
    f1   = float(f1_score(y_val, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_val, y_prob))

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Thumbs Down', 'Thumbs Up'])
    ax.set_yticklabels(['Thumbs Down', 'Thumbs Up'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve'); ax.legend(loc='lower right')
    fig.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    metrics = {
        'validation_samples': len(y_val),
        'accuracy':  round(acc,  4),
        'precision': round(prec, 4),
        'recall':    round(rec,  4),
        'f1_score':  round(f1,   4),
        'roc_auc':   round(auc,  4),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults ({len(y_val)} val samples)")
    print(f"  Accuracy  {acc:.4f}")
    print(f"  Precision {prec:.4f}")
    print(f"  Recall    {rec:.4f}")
    print(f"  F1        {f1:.4f}")
    print(f"  AUC       {auc:.4f}\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    default=DEFAULT_CSV)
    parser.add_argument('--output', default=DEFAULT_MODEL_OUTPUT)
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"Loading {args.csv}")
    images, labels = preprocess_data(args.csv, normalize=False)
    print(f"{len(images)} images  ({labels.sum()} thumbs_up / {len(labels)-labels.sum()} thumbs_down)")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=labels
    )

    X_val_prep = mobilenet_preprocess(X_val.copy())

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    datagen = get_augmentation()
    datagen.fit(X_train)
    train_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=math.ceil(len(X_train) / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val_prep, y_val),
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=PATIENCE,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                args.output, monitor='val_accuracy',
                save_best_only=True, verbose=1,
            ),
        ],
        verbose=1,
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(OUTPUTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

    plot_history(history, OUTPUTS_DIR)
    evaluate(model, X_val_prep, y_val, OUTPUTS_DIR)

    tflite_path = os.path.splitext(args.output)[0] + '.tflite'
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(tflite_path, 'wb') as f:
        f.write(converter.convert())

    print(f"Saved: {args.output}  |  {tflite_path}")


if __name__ == '__main__':
    main()
