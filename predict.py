import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = "traffic_sign_model.keras"
CLASS_FILE = "class_names.txt"

# Change this to the name of your test image
IMAGE_PATH = "test_image.jpg"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        "Model not found. Run train.py first."
    )

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Could not read the image.")

image = cv2.resize(image, (32, 32))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image, verbose=0)
index = int(np.argmax(prediction[0]))
confidence = float(prediction[0][index] * 100)

print("\n==============================")
print("   TRAFFIC SIGN RECOGNITION")
print("==============================")
print("Predicted Sign :", class_names[index])
print(f"Confidence     : {confidence:.2f}%")
print("==============================")
