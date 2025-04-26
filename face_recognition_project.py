# -*- coding: utf-8 -*-
"""
Face Recognition Project
Adapted for Local Environment (Terminal / VSCode / PyCharm).
This script performs face recognition using Eigenfaces + SVM.

"""

import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from mean_centering import MeanCentering


# Set dataset directory
dataset_dir = 'images'

# Define functions

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    image = cv2.imread(image_path)
    if image is None:
        print(f'[WARNING] Could not load image: {image_path}')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces using Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

def crop_faces(image_gray, faces, return_all=False):
    """Crop detected faces from an image."""
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                cropped_faces.append(image_gray[y:y+h, x:x+w])
                selected_faces.append((x, y, w, h))
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])  # biggest face
            cropped_faces.append(image_gray[y:y+h, x:x+w])
            selected_faces.append((x, y, w, h))
    return cropped_faces, selected_faces

def resize_and_flatten(face, face_size=(128, 128)):
    """Resize a face and flatten it into 1D array."""
    face_resized = cv2.resize(face, face_size)
    return face_resized.flatten()

# Prepare dataset
images = []
labels = []

for root, dirs, files in os.walk(dataset_dir):
    if len(files) == 0:
        continue
    for file in files:
        _, gray_image = load_image(os.path.join(root, file))
        if gray_image is None:
            continue
        images.append(gray_image)
        labels.append(os.path.basename(root))

# Feature extraction
X = []
y = []

for image, label in zip(images, labels):
    faces = detect_faces(image)
    cropped_faces, _ = crop_faces(image, faces)
    if len(cropped_faces) > 0:
        face_flattened = resize_and_flatten(cropped_faces[0])
        X.append(face_flattened)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=177, stratify=y
)

# Preprocessing: Mean Centering
from mean_centering import MeanCentering

# Build Pipeline: Mean Centering → PCA → SVM
pipe = Pipeline([
    ('mean_centering', MeanCentering()),
    ('pca', PCA(0.95)),
    ('svc', SVC(kernel='linear', probability=True))
])

# Model training
pipe.fit(X_train, y_train)

# Prediction
y_pred = pipe.predict(X_test)

# Evaluation
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

# Save the pipeline model
with open('eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Visualize Eigenfaces
n_components = len(pipe.named_steps['pca'].components_)
eigenfaces = pipe.named_steps['pca'].components_.reshape((n_components, X_train.shape[1]))

ncol = 4
nrow = (n_components + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3 * nrow), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    if i >= n_components:
        break
    ax.imshow(eigenfaces[i].reshape((128, 128)), cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
plt.tight_layout()
plt.show()

print("[INFO] Model training and saving complete.")
