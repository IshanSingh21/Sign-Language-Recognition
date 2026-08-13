# ASL Sign Language Gesture Recognition

A complete Python pipeline for recognizing American Sign Language (ASL) gestures using computer vision and machine learning. Supports real-time webcam recognition of the ASL alphabet (A–Z) and 5 common phrases.

---

## 🚀 Features

- 📷 **Real-Time Hand Tracking**: Fast and robust hand landmark detection using MediaPipe Hands (21 3D landmarks per hand).
- 🤖 **ML-Based Classification**: Compare and train multiple machine learning models (Random Forest, SVM, MLP Classifier).
- 📊 **93-Feature Vector Extraction**: Translates raw hand landmarks into scale- and location-invariant features (normalized coordinates + joint angles + fingertip distances + finger extension states).
- 🎯 **Temporal Smoothing**: Rolling window majority vote algorithm to minimize flickering and produce stable real-time predictions.
- 📝 **Interactive Data Collection Tool**: Built-in OpenCV GUI to easily gather, preview, and label hand gesture datasets.
- 💬 **Sentence Builder**: Real-time word spelling tool with auto-space insertion, deletion, and buffer clearing.
- 🖐️ **31 Gesture Classes**: Full ASL Alphabet (A–Z) plus 5 common phrases (`HELLO`, `THANK_YOU`, `YES`, `NO`, `I_LOVE_YOU`).

---

## 💻 Requirements

- **Python**: 3.10 – 3.12 (recommended)
- **Hardware**: Standard webcam
- **OS**: Windows (tested), Linux, macOS
- **Core Dependencies**:
  - `mediapipe`
  - `opencv-python`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `tensorflow` (optional, for custom deep learning extensions)

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd sign_language_recognition
   ```

2. **Create and activate a virtual environment** *(optional but recommended)*:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage — 3 Steps

### Step 1: Collect Training Data

Launch the interactive collection script to capture gesture samples from your webcam:

```bash
python collect_data.py
```

**Controls**:
| Key | Action |
| --- | --- |
| `UP` / `DOWN` | Navigate through gesture labels |
| `SPACE` | Toggle continuous auto-capture mode |
| `S` | Capture a single frame |
| `N` / `P` | Jump to Next / Previous label |
| `Q` | Quit data collection |

> 💡 **Recommendation**: Collect **200 to 500 samples per gesture class** under varied lighting conditions and slightly different hand positions/angles for maximum accuracy.

---

### Step 2: Train the Model

Train and evaluate the gesture classification models on your collected dataset:

```bash
python train_model.py
```

**What this step does**:
- Loads and preprocesses the dataset from `data/gesture_data.csv`.
- Trains and compares **Random Forest**, **Support Vector Machine (SVM)**, and **Multi-Layer Perceptron (MLP)** models.
- Evaluates model performance on validation and test sets.
- Automatically selects and saves the **best-performing model** to `models/best_model.pkl`.
- Exports evaluation logs (`models/training_report.txt`) and a high-resolution **confusion matrix plot** (`models/confusion_matrix.png`).

---

### Step 3: Run Real-Time Recognition

Run the real-time inference application with live webcam feed and sentence construction UI:

```bash
python recognize.py
```

**Controls**:
| Key | Action |
| --- | --- |
| `SPACE` | Append currently detected letter/phrase to the sentence buffer |
| `BACKSPACE` | Delete the last character in the sentence |
| `C` | Clear the entire sentence buffer |
| `Q` | Exit recognition application |

---

## 📁 Project Structure

```
sign_language_recognition/
├── config.py                 # Global configuration settings, constants, and paths
├── collect_data.py           # Interactive data collection tool with OpenCV UI
├── train_model.py            # Model training, evaluation, and serialization pipeline
├── recognize.py              # Real-time gesture recognition & interactive sentence builder
├── requirements.txt          # Python dependencies specification
├── README.md                 # Project documentation
├── data/                     # Dataset directory
│   └── gesture_data.csv      # Extracted 93-feature training data
├── models/                   # Directory for trained models and evaluation outputs
│   ├── best_model.pkl        # Top-performing trained model file
│   ├── scaler.pkl            # Feature scaling transformer
│   ├── label_encoder.pkl     # Label encoder mapping
│   ├── training_report.txt   # Detailed classification report
│   └── confusion_matrix.png  # Confusion matrix visualization
└── utils/                    # Modular utility package
    ├── __init__.py           # Utility package initialization
    ├── hand_detector.py      # MediaPipe hand detection & landmark processing wrapper
    ├── feature_extractor.py  # 93-feature vector calculation (coords, angles, distances, states)
    ├── smoothing.py          # Temporal rolling window majority voting for stability
    └── visualization.py      # OpenCV UI panels, overlays, and bounding box renderers
```

---

## ⚙️ How It Works

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│  Webcam Input   │ ─► │ MediaPipe Hands    │ ─► │ Feature Extraction  │ ─► │  ML Classifier   │
│  (Live Video)   │    │ (21 3D Landmarks)  │    │ (93-dim Vector)     │    │ (RF / SVM / MLP) │
└─────────────────┘    └────────────────────┘    └─────────────────────┘    └─────────┬────────┘
                                                                                      │
                                                                                      ▼
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ Interactive UI  │ ◄─ │ Sentence Builder   │ ◄─ │ Temporal Smoothing  │ ◄─ │ Predicted Label  │
│  (OpenCV Window)│    │ (Word Assembly)    │    │ (Rolling Vote Filter│    │ & Confidence     │
└─────────────────┘    └────────────────────┘    └─────────────────────┘    └──────────────────┘
```

1. **Hand Detection**: MediaPipe tracks 21 3D hand keypoints in real time ($x, y, z$ coordinates).
2. **Feature Extraction**: Landmark coordinates are processed into a **93-dimensional feature vector**:
   - **63 Normalized Coordinates**: Re-centered to hand wrist landmark and normalized by palm bounding scale.
   - **15 Joint Angles**: Angles at MCP, PIP, and DIP joints across all 5 fingers.
   - **10 Fingertip Pairwise Distances**: Relative spatial distances between all pairs of fingertips ($C(5,2) = 10$).
   - **5 Finger Extension States**: Binary/continuous extension ratios indicating whether each finger is open or curled.
3. **Classification**: The feature vector is passed to the trained classifier (Random Forest / SVM / MLP) to generate probability predictions across all 31 classes.
4. **Temporal Smoothing**: A rolling window buffer aggregates predictions over consecutive frames and applies majority voting to prevent flickering.

---

## 🖐️ Supported Gestures (31 Classes)

| Category | Gestures / Labels | Count |
| --- | --- | --- |
| **ASL Alphabet** | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` | 26 |
| **Common Phrases** | `HELLO`, `THANK_YOU`, `YES`, `NO`, `I_LOVE_YOU` | 5 |

---

## 💡 Tips for Better Accuracy

1. **Good Lighting**: Ensure your hand is evenly illuminated without harsh backlighting or heavy shadows.
2. **Plain Background**: Use a clean, non-distracting background to avoid false hand detections.
3. **Consistent Hand Distance & Position**: Keep your hand within 1 to 3 feet of the camera for optimal MediaPipe landmark tracking.
4. **Varied Training Data**: When collecting samples, slowly rotate your hand slightly and capture data under different lighting angles.
5. **Sufficient Dataset Size**: Collect at least **200 to 500 quality samples per gesture class**.

---

## 🔍 Troubleshooting

- **MediaPipe Import Error**: Ensure you are using Python 3.10–3.12. MediaPipe may have compatibility issues with newer Python versions (e.g. 3.13+).
- **Webcam Fails to Open**: If `CAMERA_INDEX = 0` fails, try changing `CAMERA_INDEX` to `1` or `2` in `config.py`.
- **Low Classification Accuracy**: Ensure enough dataset samples have been collected across all classes, and re-run `python train_model.py`.
- **DLL Initialization Failure (Windows)**: Install the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
