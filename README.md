# 🤟 ASL Sign Language Gesture Recognition

 Real-time American Sign Language (ASL) gesture recognition using computer vision and machine learning.


## 🚀 Live Demo

**[Try the deployed application →](https://sign-language-recognition-cpqc.onrender.com)**

Experience the project directly in your browser without installing Python or configuring the machine-learning environment.

> **Note:** Camera permissions may be required for real-time gesture recognition.

---

## 📌 About the Project

This project is a real-time **American Sign Language (ASL) gesture recognition system** that combines computer vision and machine learning to recognize hand gestures through a camera.

The system uses **MediaPipe Hands** to detect 21 3D hand landmarks, transforms them into a **93-dimensional feature vector**, and uses machine-learning classifiers to recognize predefined ASL gestures.

The project supports **31 gesture classes**:

* 🔤 26 ASL alphabet gestures (`A–Z`)
* 💬 5 common phrases (`HELLO`, `THANK_YOU`, `YES`, `NO`, `I_LOVE_YOU`)

The project also includes temporal prediction smoothing and an interactive sentence-building system.

---

## ✨ Key Features

* 📷 **Real-Time Hand Tracking** using MediaPipe Hands
* 🤖 **Machine Learning Classification** using Random Forest, SVM, and MLP
* 📊 **93-Dimensional Feature Extraction**
* 🎯 **Temporal Smoothing** for stable predictions
* 💬 **Interactive Sentence Builder**
* 🖐️ **31 Gesture Classes**
* 🌐 **Browser-Based Web Application**
* 📈 **Model Evaluation & Confusion Matrix**
* 🧪 **Custom Dataset Collection Pipeline**

---

## 🧠 How It Works

```text
Webcam
   │
   ▼
MediaPipe Hand Detection
   │
   │ 21 3D Landmarks
   ▼
Feature Extraction
   │
   │ 93 Features
   ▼
Machine Learning Classifier
   │
   │ Random Forest / SVM / MLP
   ▼
Gesture Prediction
   │
   ▼
Temporal Smoothing
   │
   ▼
Sentence Builder
```

### Processing Pipeline

1. **Hand Detection**
   MediaPipe detects 21 3D landmarks from the user's hand.

2. **Feature Extraction**
   Landmark coordinates are transformed into a 93-dimensional feature vector.

3. **Classification**
   The feature vector is passed to the trained machine-learning classifier.

4. **Temporal Smoothing**
   Predictions from consecutive frames are combined using rolling-window majority voting to reduce flickering.

5. **Sentence Building**
   Recognized letters and phrases can be added to an interactive sentence buffer.

---

## 📊 Feature Engineering

The 93-dimensional feature vector consists of:

| Feature                         | Dimensions |
| ------------------------------- | ---------: |
| Normalized landmark coordinates |         63 |
| Joint angles                    |         15 |
| Fingertip pairwise distances    |         10 |
| Finger extension features       |          5 |
| **Total**                       |     **93** |

The landmark coordinates are normalized relative to the hand position and scale to make the representation more robust to changes in hand location.

---

## 🖐️ Supported Gestures

### ASL Alphabet — 26 Classes

`A` `B` `C` `D` `E` `F` `G` `H` `I` `J` `K` `L` `M`

`N` `O` `P` `Q` `R` `S` `T` `U` `V` `W` `X` `Y` `Z`

### Common Phrases — 5 Classes

* `HELLO`
* `THANK_YOU`
* `YES`
* `NO`
* `I_LOVE_YOU`

**Total: 31 gesture classes**

---

## 🛠️ Tech Stack

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Core programming                    |
| OpenCV       | Computer vision & camera processing |
| MediaPipe    | Hand landmark detection             |
| NumPy        | Numerical computation               |
| Pandas       | Dataset processing                  |
| Scikit-learn | Machine learning                    |
| Matplotlib   | Visualization                       |
| Seaborn      | Model evaluation                    |
| Flask/Web    | Web application                     |
| Render       | Deployment                          |

---

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── data/
│   └── gesture_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── training_report.txt
│   └── confusion_matrix.png
│
├── utils/
│   ├── hand_detector.py
│   ├── feature_extractor.py
│   ├── smoothing.py
│   └── visualization.py
│
├── web/
│   └── Web application files
│
├── collect_data.py
├── config.py
├── train_model.py
├── recognize.py
├── live_recognize.py
├── run_web.py
├── requirements.txt
└── README.md
```

---

## 💻 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/IshanSingh21/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

To collect your own gesture dataset:

```bash
python collect_data.py
```

Then train and compare the machine-learning models:

```bash
python train_model.py
```

The training pipeline:

* Loads the gesture dataset
* Preprocesses the feature vectors
* Trains Random Forest, SVM, and MLP models
* Evaluates model performance
* Selects the best-performing model
* Saves the trained model and preprocessing artifacts

---

## 🖥️ Run Real-Time Recognition Locally

```bash
python recognize.py
```

### Controls

| Key         | Action                |
| ----------- | --------------------- |
| `SPACE`     | Add detected gesture  |
| `BACKSPACE` | Delete last character |
| `C`         | Clear sentence        |
| `Q`         | Exit                  |

---

## 🌐 Run the Web Application Locally

```bash
python run_web.py
```

After starting the server, open the local URL displayed by the application.

Alternatively, use the deployed version:

### 👉 [Open Live Demo](https://sign-language-recognition-cpqc.onrender.com)

---

## 📈 Model Evaluation

The project evaluates multiple machine-learning approaches:

* Random Forest
* Support Vector Machine (SVM)
* Multi-Layer Perceptron (MLP)

The training pipeline generates:

* Classification reports
* Model comparison results
* Confusion matrix
* Trained model
* Feature scaler
* Label encoder

---

## 🎯 Tips for Better Recognition

For improved recognition accuracy:

1. Use consistent and sufficient lighting.
2. Keep your hand clearly visible.
3. Avoid highly cluttered backgrounds.
4. Maintain a consistent distance from the camera.
5. Collect diverse training samples.
6. Include variations in hand position and orientation.
7. Use enough samples for every gesture class.

The existing project recommends collecting approximately **200–500 samples per gesture class** when creating a custom dataset.

---

## ⚠️ Limitations

This project focuses on **isolated gesture recognition**, rather than complete continuous ASL translation.

Recognition can be affected by:

* Lighting conditions
* Camera quality
* Background complexity
* Hand orientation
* Similarity between gestures
* Training dataset quality

The system recognizes predefined gesture classes and does not currently perform full natural-language translation of continuous sign language.

---

## 🔮 Future Improvements

* [ ] Continuous sign-language sentence recognition
* [ ] Word-level gesture recognition
* [ ] Text-to-speech output
* [ ] Support for additional sign languages
* [ ] Larger and more diverse datasets
* [ ] Deep-learning-based gesture recognition
* [ ] Mobile application
* [ ] Multi-hand gesture recognition
* [ ] Improved browser-based inference
* [ ] Better robustness across lighting conditions

---

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Commit your changes.
5. Push your branch.
6. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Ishan Singh**

GitHub: [@IshanSingh21](https://github.com/IshanSingh21)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐.

### 🚀 Try It Yourself

**[🌐 Live Demo](https://sign-language-recognition-cpqc.onrender.com)**

**[💻 Source Code](https://github.com/IshanSingh21/Sign-Language-Recognition)**
