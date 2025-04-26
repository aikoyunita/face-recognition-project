# Face Recognition Project

This project implements a basic **face detection** and **face recognition** system using OpenCV, PCA (Principal Component Analysis), and SVM (Support Vector Machine).  
It was developed as part of a learning assignment on face recognition techniques and real-time testing using a webcam.

---

## 📂 Project Structure
```
face-recognition-project/
├── images/                   # Folder containing the training images (categorized by person)
├── face_recognition_project.py    # Script to train the model and save the pipeline
├── realtime_face_recognition.py   # Script to perform real-time face detection and recognition
├── mean_centering.py          # Custom MeanCentering transformer for preprocessing
├── eigenface_pipeline.pkl     # Trained model (generated after training)
├── requirements.txt           # List of Python dependencies
└── README.md                  # This README file
```

---

## ⚙️ How to Run the Project

### 1. Install Dependencies
Make sure you have Python installed. Then inside your project folder, run:

```bash
pip install -r requirements.txt
```

### 2. Train the Model
To train the model and generate the `eigenface_pipeline.pkl`, run:

```bash
python face_recognition_project.py
```

This script will:
- Load the dataset (`images/` folder).
- Train the model (Mean Centering → PCA → SVM).
- Save the model to `eigenface_pipeline.pkl`.

### 3. Run Real-Time Face Recognition
To perform real-time face detection and recognition using your webcam:

```bash
python realtime_face_recognition.py
```

Press `Q` on the keyboard to exit the webcam window.

---

## 📋 Important Notes
- The `images/` folder should be organized by **subfolders**, each containing images of one person.  
  Example:
  ```
  images/
    ├── Aiko/
    ├── Raihan/
    ├── Laura_Bush/
    ├── Vladimir_Putin/
    └── etc.
  ```
- During real-time recognition:
  - If the face is recognized, it shows the name and confidence score.
  - If not, it shows "Unknown".
- Make sure the lighting is good and the face is clearly visible for better results.

---

## 🛠️ Technologies Used
- Python 3
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib

---
