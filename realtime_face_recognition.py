import cv2
import pickle
import numpy as np

# Load trained model
print("[INFO] Loading trained model...")
with open('eigenface_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract label names
label_names = np.unique(model.named_steps['svc'].classes_)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

# Set target size for face resizing
face_size = (128, 128)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, face_size).flatten().reshape(1, -1)

        # Preprocess face (Mean Centering -> PCA)
        preprocessed_face = model.named_steps['mean_centering'].transform(face_resized)
        preprocessed_face = model.named_steps['pca'].transform(preprocessed_face)

        # Predict
        pred = model.named_steps['svc'].predict(preprocessed_face)
        confidence = model.named_steps['svc'].decision_function(preprocessed_face)

        pred_label = pred[0] if len(pred) > 0 else "Unknown"
        confidence_score = float(np.max(confidence))

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f'{pred_label} ({confidence_score:.2f})'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()