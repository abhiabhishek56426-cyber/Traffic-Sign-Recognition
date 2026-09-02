import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

MODEL_PATH = "traffic_sign_model.keras"
CLASS_FILE = "class_names.txt"

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("650x600")

        if not Path(MODEL_PATH).exists():
            messagebox.showerror(
                "Model Missing",
                "Run train.py first to create traffic_sign_model.keras."
            )
            root.destroy()
            return

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_FILE, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        tk.Label(
            root,
            text="TRAFFIC SIGN RECOGNITION",
            font=("Arial", 20, "bold")
        ).pack(pady=20)

        self.image_label = tk.Label(root, text="Select a traffic sign image")
        self.image_label.pack(pady=10)

        tk.Button(
            root,
            text="SELECT IMAGE",
            command=self.select_image,
            font=("Arial", 12, "bold")
        ).pack(pady=15)

        self.result_label = tk.Label(
            root,
            text="Prediction: -\nConfidence: -",
            font=("Arial", 16)
        )
        self.result_label.pack(pady=20)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        # Show image
        pil_image = Image.open(path)
        pil_image.thumbnail((400, 300))
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.tk_image, text="")

        # Predict
        image = cv2.imread(path)
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = self.model.predict(image, verbose=0)
        index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][index] * 100)

        self.result_label.configure(
            text=f"Prediction: {self.class_names[index]}\n"
                 f"Confidence: {confidence:.2f}%"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()
