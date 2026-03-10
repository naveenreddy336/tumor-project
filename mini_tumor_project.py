import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.applications import DenseNet121, ResNet101
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1️⃣ Load Pretrained Models
# -----------------------------
print("Loading Deep Learning Models...")

densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
resnet = ResNet101(weights='imagenet', include_top=False, pooling='avg')

# -----------------------------
# 2️⃣ Feature Extraction
# -----------------------------
print("Extracting Features...")

X = []
y = []

base_path = "dataset/train"   # dataset/train/normal & dataset/train/tumor

for label, folder in enumerate(['normal', 'tumor']):
    folder_path = os.path.join(base_path, folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        f1 = densenet.predict(img, verbose=0)
        f2 = resnet.predict(img, verbose=0)

        fused = np.concatenate((f1[0], f2[0]))
        X.append(fused)
        y.append(label)

X = np.array(X)
y = np.array(y)

# -----------------------------
# 3️⃣ Train Model (SVM)
# -----------------------------
print("Training SVM Model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(probability=True)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)

# Save model
joblib.dump(model, "tumor_model.pkl")
print("Model Saved Successfully!")

# -----------------------------
# 4️⃣ Predict New Image
# -----------------------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    f1 = densenet.predict(img, verbose=0)
    f2 = resnet.predict(img, verbose=0)

    fused = np.concatenate((f1[0], f2[0]))

    result = model.predict([fused])

    if result[0] == 1:
        print("Prediction: Tumor Detected")
    else:
        print("Prediction: Normal")

# Example
predict_image("dataset/train/normal/Te-no_2.jpg")
