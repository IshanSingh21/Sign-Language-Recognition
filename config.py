"""
config.py — Global configuration for ASL Sign Language Gesture Recognition.

Contains all constants, paths, labels, and hyperparameters used across
data collection, model training, and real-time inference.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_FILE = DATA_DIR / "gesture_data.csv"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Gesture Labels (31 classes)
# ──────────────────────────────────────────────────────────────────────
ALPHABET_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
PHRASE_LABELS = ["HELLO", "THANK_YOU", "YES", "NO", "I_LOVE_YOU"]
ALL_LABELS = ALPHABET_LABELS + PHRASE_LABELS
NUM_CLASSES = len(ALL_LABELS)

# ──────────────────────────────────────────────────────────────────────
# MediaPipe Settings
# ──────────────────────────────────────────────────────────────────────
MP_MAX_HANDS = 1                # Max number of hands to detect
MP_DETECTION_CONFIDENCE = 0.7   # Min confidence for hand detection
MP_TRACKING_CONFIDENCE = 0.6    # Min confidence for hand tracking

# ──────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21              # MediaPipe hand landmarks per hand
COORDS_PER_LANDMARK = 3         # x, y, z
NUM_RAW_FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

# Derived features
NUM_JOINT_ANGLES = 15           # 3 angles per finger × 5 fingers
NUM_FINGERTIP_DISTANCES = 10    # C(5,2) pairwise fingertip distances
NUM_FINGER_STATES = 5           # Extended/curled per finger

TOTAL_FEATURES = (
    NUM_RAW_FEATURES +
    NUM_JOINT_ANGLES +
    NUM_FINGERTIP_DISTANCES +
    NUM_FINGER_STATES
)  # 93

# ──────────────────────────────────────────────────────────────────────
# Data Collection
# ──────────────────────────────────────────────────────────────────────
CAPTURE_FPS = 10                # Samples per second during auto-capture
MIN_SAMPLES_PER_LABEL = 200     # Recommended minimum per gesture
TARGET_SAMPLES_PER_LABEL = 500  # Target for robust training

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.15                # Test set ratio
VAL_SIZE = 0.15                 # Validation set ratio (from remaining)
RANDOM_STATE = 42               # Random seed for reproducibility

# MLP Architecture
MLP_HIDDEN_LAYERS = (256, 128, 64)
MLP_MAX_ITER = 500
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 100

# ──────────────────────────────────────────────────────────────────────
# Inference / Recognition
# ──────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75     # Min confidence to display prediction
SMOOTHING_WINDOW = 10           # Number of frames for majority vote
SENTENCE_HOLD_FRAMES = 15       # Frames a letter must be stable before auto-add

# ──────────────────────────────────────────────────────────────────────
# Webcam
# ──────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0                # Default camera
CAMERA_WIDTH = 1280             # Capture width
CAMERA_HEIGHT = 720             # Capture height

# ──────────────────────────────────────────────────────────────────────
# UI Colors (BGR format for OpenCV)
# ──────────────────────────────────────────────────────────────────────
COLOR_PRIMARY = (255, 165, 0)       # Orange-ish
COLOR_SECONDARY = (200, 130, 255)   # Light purple
COLOR_SUCCESS = (80, 220, 100)      # Green
COLOR_DANGER = (60, 60, 255)        # Red
COLOR_WARNING = (0, 200, 255)       # Yellow
COLOR_TEXT = (255, 255, 255)        # White
COLOR_TEXT_DIM = (180, 180, 180)    # Gray
COLOR_BG_DARK = (30, 30, 30)       # Dark background
COLOR_BG_PANEL = (40, 40, 50)      # Panel background
COLOR_RECORDING = (0, 0, 255)      # Red for recording indicator
COLOR_CONFIDENCE_HIGH = (80, 220, 100)  # Green
COLOR_CONFIDENCE_MED = (0, 200, 255)    # Yellow
COLOR_CONFIDENCE_LOW = (60, 60, 255)    # Red

# ──────────────────────────────────────────────────────────────────────
# Model File Paths
# ──────────────────────────────────────────────────────────────────────
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.txt"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
