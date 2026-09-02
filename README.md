# Traffic Sign Recognition Using CNN

A simple mini-project for recognizing traffic sign images using a
Convolutional Neural Network (CNN) in Python.

## 1. Project structure

Traffic_Sign_Recognition/
│
├── dataset/
│   ├── train/
│   │   ├── Stop/
│   │   ├── Speed_Limit/
│   │   ├── No_Entry/
│   │   └── ...
│   └── test/
│       ├── Stop/
│       ├── Speed_Limit/
│       ├── No_Entry/
│       └── ...
│
├── train.py
├── predict.py
├── gui.py
├── requirements.txt
└── README.md

The folder names inside train and test are the class names.

## 2. Install Python

Install Python 3 from:
https://www.python.org/downloads/

During installation, enable "Add Python to PATH".

## 3. Open this folder in VS Code

In VS Code:
File -> Open Folder -> Traffic_Sign_Recognition

Then open:
Terminal -> New Terminal

## 4. Install packages

Run:

pip install -r requirements.txt

If "pip" does not work, try:

py -m pip install -r requirements.txt

## 5. Add your dataset

For this simple version, organize images into folders.

Example:

dataset/train/Stop/
dataset/train/Speed_Limit/
dataset/train/No_Entry/

and:

dataset/test/Stop/
dataset/test/Speed_Limit/
dataset/test/No_Entry/

Use the SAME class folders in train and test.

For a very small college mini-project, start with 5 classes:
- Stop
- Speed_Limit
- No_Entry
- Turn_Left
- Turn_Right

Try to collect many different images per class. More images generally improve
the model.

## 6. Train the model

Run:

python train.py

or:

py train.py

After training, these files are created:

traffic_sign_model.keras
class_names.txt
accuracy.png

## 7. Predict one image

Copy a traffic sign image into the project root and name it:

test_image.jpg

Then run:

python predict.py

The output will look like:

Predicted Sign : Stop
Confidence     : 95.23%

## 8. Run the GUI

Install Pillow if not already installed:

pip install pillow

Then run:

python gui.py

A window opens where you can select an image and see the prediction.

## 9. Important note

This project expects a folder-based image dataset. The original GTSRB dataset
is commonly distributed in other formats as well, so do not simply copy a
CSV-only GTSRB download into dataset/. If you want to use the original GTSRB
dataset, convert/organize its images into the folder structure above first.

## 10. Mini-project explanation

Input image
    ↓
Resize to 32 x 32
    ↓
Normalize pixels
    ↓
CNN layers
    ↓
Feature extraction
    ↓
Softmax classification
    ↓
Traffic sign name + confidence
