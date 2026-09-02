import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Load images from folders.
# Folder names become the class names.
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names
print("Classes:", class_names)

# Improve input pipeline speed
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# Save model and class names
model.save("traffic_sign_model.keras")

with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("\nTraining complete!")
print("Saved: traffic_sign_model.keras")
print("Saved: class_names.txt")

# Accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Traffic Sign Recognition - Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy.png")
plt.show()
