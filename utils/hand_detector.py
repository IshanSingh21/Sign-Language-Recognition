"""
hand_detector.py — MediaPipe hand landmark detection wrapper.

Uses the MediaPipe Tasks API (HandLandmarker) for detecting hands and
extracting 21 3D landmarks from webcam frames.

Compatible with mediapipe >= 1.0.0.
"""

import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
    HandLandmarksConnections,
    drawing_utils,
    drawing_styles,
)


# Default model path (relative to project root)
_DEFAULT_MODEL_PATH = str(
    (Path(__file__).parent.parent / "models" / "hand_landmarker.task").resolve()
)


class HandDetector:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) for hand landmark detection.

    Attributes:
        landmarker: MediaPipe HandLandmarker instance.
    """

    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_path=None,
    ):
        """
        Initialize the hand detector.

        Args:
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            model_path: Path to hand_landmarker.task file.
                        Defaults to models/hand_landmarker.task.
        """
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
                "Place it in the models/ directory."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def detect(self, frame):
        """
        Detect hands in a BGR frame and return the result.

        Args:
            frame: BGR image (numpy array) from OpenCV.

        Returns:
            HandLandmarkerResult containing:
                - hand_landmarks: List of NormalizedLandmark lists.
                - handedness: List of handedness classifications.
            Returns None if detection fails.
        """
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Increment timestamp (VIDEO mode requires monotonically increasing timestamps)
        self._frame_timestamp_ms += 33  # ~30 fps

        try:
            results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return None

    def get_landmarks(self, results):
        """
        Extract landmark coordinates from HandLandmarkerResult.

        Args:
            results: HandLandmarkerResult from detect().

        Returns:
            List of numpy arrays, each of shape (21, 3) containing
            [x, y, z] for each landmark. Returns empty list if no hands.
        """
        all_landmarks = []

        if results and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32,
                )
                all_landmarks.append(landmarks)

        return all_landmarks

    def draw_landmarks(self, frame, results):
        """
        Draw detected hand landmarks on the frame.

        Args:
            frame: BGR image to draw on (modified in-place).
            results: HandLandmarkerResult from detect().

        Returns:
            frame: The frame with landmarks drawn on it.
        """
        if results and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # MediaPipe 1.0.0 draw_landmarks accepts the landmark list directly
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

        return frame

    def get_handedness(self, results):
        """
        Get the handedness (left/right) of detected hands.

        Args:
            results: HandLandmarkerResult from detect().

        Returns:
            List of strings ('Left' or 'Right') for each detected hand.
        """
        handedness = []
        if results and results.handedness:
            for hand_class_list in results.handedness:
                if hand_class_list:
                    label = hand_class_list[0].category_name
                    handedness.append(label)
        return handedness

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.release()
        except Exception:
            pass
