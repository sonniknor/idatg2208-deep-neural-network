# -*- coding: utf-8 -*-
"""IDATG2208 - Mandatory Assignment 3

Run with: python3 assignment3.py

You need tensorflow, tensorflow-datasets and matplotlib installed.
I do all six exercises in one script. Each one prints its test accuracy and
saves a plot of the training and validation loss in plots/. At the end I also
write all the numbers to results.json so I can copy them into the report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # I save the figures to files so the script doesn't stop to show windows
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
VAL_SPLIT = 0.1
results = {}


def set_seed():
    tf.keras.utils.set_random_seed(SEED)


def plot_loss(history, title):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.split(':')[0].lower().replace(' ', '')}.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    best = int(np.argmin(history.history["val_loss"])) + 1
    print(f"{title}: trained {len(history.history['loss'])} epochs, "
          f"best val_loss at epoch {best}")


def early_stop():
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)


def mc_predict(model, X, passes=20):
    """Run the model `passes` times with dropout turned on (training=True).
    Gives an array of shape (passes, N, classes)."""
    return np.stack([model(X, training=True).numpy() for _ in range(passes)])


# ---------------------------------------------------------------------------
# Exercise 1: DNN on Fashion-MNIST
# ---------------------------------------------------------------------------
set_seed()
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train[:2000].astype("float32") / 255.0
y_train = y_train[:2000]
X_test = X_test[:400].astype("float32") / 255.0
y_test = y_test[:400]

model = keras.Sequential([layers.Input(shape=(28, 28)), layers.Flatten()] + [
    layers.Dense(100, activation="elu", kernel_initializer="he_normal")
    for _ in range(3)
] + [layers.Dense(10, activation="softmax")])
model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=50, batch_size=32,
                 validation_split=VAL_SPLIT, callbacks=[early_stop()], verbose=2)
plot_loss(hist, "Exercise 1: Fashion-MNIST DNN")
acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
results["Q1.1"] = acc
print(f"Q1.1 Test accuracy: {acc:.2f}%")

# ---------------------------------------------------------------------------
# Exercise 2: DNN on CIFAR-100 (coarse labels)
# ---------------------------------------------------------------------------
set_seed()
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(
    label_mode="coarse")
X_train = X_train[:5000].astype("float32") / 255.0
y_train = y_train[:5000].ravel()
X_test = X_test[:1000].astype("float32") / 255.0
y_test = y_test[:1000].ravel()

model = keras.Sequential([layers.Input(shape=(32, 32, 3)), layers.Flatten()] + [
    layers.Dense(256, activation="elu", kernel_initializer="he_normal")
    for _ in range(4)
] + [layers.Dense(20, activation="softmax")])
model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=50, batch_size=128,
                 validation_split=VAL_SPLIT, callbacks=[early_stop()], verbose=2)
plot_loss(hist, "Exercise 2: CIFAR-100 (coarse) DNN")
acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
results["Q2.1"] = acc
print(f"Q2.1 Test accuracy: {acc:.2f}%")

# ---------------------------------------------------------------------------
# Exercise 3: Alpha Dropout + MC Dropout on Fashion-MNIST
# ---------------------------------------------------------------------------
set_seed()
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train[:2000].astype("float32").reshape(-1, 784)
y_train = y_train[:2000]
X_test = X_test[:400].astype("float32").reshape(-1, 784)
y_test = y_test[:400]

# Standardize with the mean/std of the training set only (also used on the test set)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-7
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

hidden = []
for _ in range(3):
    hidden += [layers.Dense(100, activation="selu",
                            kernel_initializer="lecun_normal"),
               layers.AlphaDropout(0.1)]
model = keras.Sequential([layers.Input(shape=(784,))] + hidden +
                         [layers.Dense(10, activation="softmax")])
model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=50, batch_size=32,
                 validation_split=VAL_SPLIT, callbacks=[early_stop()], verbose=2)
plot_loss(hist, "Exercise 3: Alpha Dropout network")
acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
results["Q3.1"] = acc
print(f"Q3.1 Test accuracy (Alpha Dropout): {acc:.2f}%")

probs = mc_predict(model, X_test, 20).mean(axis=0)
mc_acc = np.mean(probs.argmax(axis=1) == y_test) * 100
results["Q3.2"] = mc_acc
print(f"Q3.2 MC Dropout accuracy (20 passes): {mc_acc:.2f}%")

# ---------------------------------------------------------------------------
# Exercise 4: Transfer learning (MobileNetV2) on EuroSAT
# ---------------------------------------------------------------------------
def ds_to_numpy(ds):
    xs, ys = [], []
    for x, y in tfds.as_numpy(ds):
        xs.append(x)
        ys.append(y)
    return np.stack(xs).astype("float32") / 255.0, np.array(ys)


def exercise4():
    set_seed()
    (ds_train, ds_test) = tfds.load(
        "eurosat/rgb", split=["train[:2000]", "train[2000:2500]"],
        as_supervised=True, shuffle_files=False)
    X_train, y_train = ds_to_numpy(ds_train)
    X_test, y_test = ds_to_numpy(ds_test)
    print(X_train.shape, X_test.shape)

    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(64, 64, 3))
    base.trainable = False
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(X_train, y_train, epochs=5, batch_size=32,
                     validation_split=VAL_SPLIT, verbose=2)
    plot_loss(hist, "Exercise 4: EuroSAT transfer learning")
    acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
    results["Q4.1"] = acc
    print(f"Q4.1 Test accuracy: {acc:.2f}%")


import socket


def use_eurosat_mirror_if_needed():
    """tfds normally downloads EuroSAT from madm.dfki.de, but that server
    timed out for me. If it can't be reached I make tfds download the same
    dataset from Zenodo instead."""
    try:
        socket.create_connection(("madm.dfki.de", 80), timeout=10).close()
        return
    except OSError:
        print("madm.dfki.de unreachable -> using Zenodo mirror for EuroSAT")
    for cfg in tfds.builder_cls("eurosat").BUILDER_CONFIGS:
        if cfg.name == "rgb":
            cfg.download_url = ("https://zenodo.org/records/7711810/files/"
                                "EuroSAT_RGB.zip")
            cfg.subdir = "EuroSAT_RGB"


use_eurosat_mirror_if_needed()
exercise4()

# ---------------------------------------------------------------------------
# KMNIST loader (Exercises 5 and 6)
# ---------------------------------------------------------------------------
def load_kmnist():
    ds_tr, ds_te = tfds.load("kmnist", split=["train[:3000]", "test[:500]"],
                             as_supervised=True, shuffle_files=False)
    Xtr, ytr = ds_to_numpy(ds_tr)
    Xte, yte = ds_to_numpy(ds_te)
    return Xtr, ytr, Xte, yte  # images are (N, 28, 28, 1) and already scaled to [0,1]


# ---------------------------------------------------------------------------
# Exercise 5: Deeper CNN on KMNIST
# ---------------------------------------------------------------------------
set_seed()
X_train, y_train, X_test, y_test = load_kmnist()
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=15, batch_size=32,
                 validation_split=VAL_SPLIT, verbose=2)
plot_loss(hist, "Exercise 5: Deeper CNN on KMNIST")
acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
results["Q5.1"] = acc
print(f"Q5.1 Test accuracy: {acc:.2f}%")

# ---------------------------------------------------------------------------
# Exercise 6: CNN + SGD + MC Dropout + epistemic uncertainty
# ---------------------------------------------------------------------------
set_seed()
X_train, y_train, X_test, y_test = load_kmnist()
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=15, batch_size=32,
                 validation_split=VAL_SPLIT, verbose=2)
plot_loss(hist, "Exercise 6: CNN with SGD on KMNIST")

acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
results["Q6.1"] = acc
print(f"Q6.1 Plain test accuracy: {acc:.2f}%")

passes = mc_predict(model, X_test, 20)          # shape (20 passes, N samples, 10 classes)
mc_acc = np.mean(passes.mean(axis=0).argmax(axis=1) == y_test) * 100
results["Q6.2"] = mc_acc
print(f"Q6.2 MC Dropout accuracy (20 passes): {mc_acc:.2f}%")

# Epistemic uncertainty: variance over the 20 passes for each class, then the
# average over the 10 classes, then the average over all test samples
uncertainty = passes.var(axis=0).mean(axis=1).mean()
results["Q6.3"] = round(float(uncertainty), 3)
print(f"Q6.3 Average epistemic uncertainty: {uncertainty:.3f}")

# ---------------------------------------------------------------------------
print("\n===== SUMMARY =====")
for k, v in results.items():
    print(f"{k}: {v:.3f}" if k == "Q6.3" else f"{k}: {v:.2f}%")
with open("results.json", "w") as f:
    json.dump({k: round(float(v), 3) for k, v in results.items()}, f, indent=2)
