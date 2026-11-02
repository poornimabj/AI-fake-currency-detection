"""Train a genuine-reference MobileNetV2 embedding anomaly model."""
import argparse
import cv2
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and cv2.imread(str(p)) is not None
    )


def load_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, SIZE).astype("float32")


def build_extractor():
    base = keras.applications.MobileNetV2(
        input_shape=SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = keras.Input(shape=SIZE + (3,))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="reference_embedding")(x)
    model = keras.Model(inputs, x, name="smartrupee_reference_extractor")
    model.compile()
    return model


def augmented_image(image, rng):
    angle = rng.uniform(-5, 5)
    matrix = cv2.getRotationMatrix2D(
        (SIZE[0] / 2, SIZE[1] / 2), angle, rng.uniform(0.94, 1.06)
    )
    result = cv2.warpAffine(image.astype(np.uint8), matrix, SIZE)
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
    return np.clip(result.astype("float32") + rng.uniform(-12, 12), 0, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--min-genuine", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    genuine = image_files(Path(args.data) / "train" / "genuine")
    if len(genuine) < args.min_genuine:
        raise ValueError(
            f"Reference training needs at least {args.min_genuine} readable genuine images; found {len(genuine)}."
        )

    extractor = build_extractor()
    rng = np.random.default_rng(args.seed)
    images = [load_rgb(p) for p in genuine]
    augmented = []
    for image in images:
        for _ in range(3):
            augmented.append(augmented_image(image, rng))

    embeddings = extractor.predict(np.stack(images), batch_size=8, verbose=0)
    augmented_embeddings = extractor.predict(np.stack(augmented), batch_size=8, verbose=0)
    all_embeddings = np.vstack([embeddings, augmented_embeddings]).astype("float32")
    mean = all_embeddings.mean(axis=0)
    scale = np.maximum(all_embeddings.std(axis=0), 1e-6).astype("float32")
    standardized = (all_embeddings - mean) / scale

    latent_dim = min(128, max(8, standardized.shape[1] // 4))
    inputs = keras.Input(shape=(standardized.shape[1],))
    encoded = layers.Dense(latent_dim, activation="relu")(inputs)
    outputs = layers.Dense(standardized.shape[1], activation="linear")(encoded)
    anomaly = keras.Model(inputs, outputs, name="genuine_reference_anomaly_autoencoder")
    anomaly.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    anomaly.fit(
        standardized, standardized, epochs=args.epochs,
        batch_size=min(16, len(standardized)), shuffle=True, verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )],
    )

    reconstructed = anomaly.predict(standardized, verbose=0)
    reconstruction_error = np.mean(np.square(reconstructed - standardized), axis=1)
    distances = np.sqrt(np.sum(np.square(standardized), axis=1))
    # Augmented images are out-of-sample calibration examples, so the 95th
    # percentile can reject normal variations from a small reference set.
    # Keep a conservative percentile and margin for genuine-only training.
    # With a small genuine-only set, reconstruction values vary substantially
    # between the original photographs and synthetic alignment variations.
    # A 99th-percentile floor with a conservative margin avoids false unusual
    # flags while the reference set remains limited.
    reconstruction_threshold = float(np.percentile(reconstruction_error, 99) * 4.75)
    distance_threshold = float(np.percentile(distances, 99) * 1.25)

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    extractor.save("models/genuine_feature_extractor.keras")
    anomaly.save("models/genuine_reference_anomaly.keras")
    np.savez(
        "models/genuine_reference_distribution.npz",
        mean=mean,
        scale=scale,
        reference_embeddings=embeddings.astype("float32"),
        reconstruction_threshold=np.array([reconstruction_threshold]),
        distance_threshold=np.array([distance_threshold]),
    )

    summary = {
        "model_type": "genuine_reference_anomaly_detection",
        "base_model": "MobileNetV2 ImageNet",
        "genuine_reference_count": len(genuine),
        "reference_files": [str(p.relative_to(Path(args.data))) for p in genuine],
        "embedding_dimension": int(embeddings.shape[1]),
        "reconstruction_threshold": reconstruction_threshold,
        "distance_threshold": distance_threshold,
        "counterfeit_detection_claim": False,
        "interpretation": (
            "Consistent means visually close to the available genuine-reference distribution. "
            "Unusual means a visual pattern outside that distribution; it does not prove counterfeiting."
        ),
    }
    Path("models/genuine_reference_distribution.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    keras.models.load_model("models/genuine_feature_extractor.keras")
    keras.models.load_model("models/genuine_reference_anomaly.keras")
    print("Saved models/genuine_feature_extractor.keras")
    print("Saved models/genuine_reference_anomaly.keras")
    print("Saved models/genuine_reference_distribution.npz")
    print("Reference feature extractor and anomaly model loaded successfully.")


if __name__ == "__main__":
    main()
