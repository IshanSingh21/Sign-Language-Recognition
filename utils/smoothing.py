"""
smoothing.py — Temporal prediction smoothing for stable real-time output.

Uses a rolling window with majority vote to eliminate prediction flickering
during real-time gesture recognition.
"""

from collections import deque, Counter
import numpy as np


class PredictionSmoother:
    """
    Smooths predictions over a sliding window using majority vote.
    
    This eliminates rapid flickering between predictions caused by
    frame-to-frame noise in hand landmark detection.
    """

    def __init__(self, window_size=10):
        """
        Initialize the smoother.

        Args:
            window_size: Number of recent predictions to consider.
        """
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)

    def add(self, label, confidence):
        """
        Add a new prediction to the buffer.

        Args:
            label: Predicted class label (string).
            confidence: Prediction confidence (float, 0-1).
        """
        self.predictions.append(label)
        self.confidences.append(confidence)

    def get_smoothed(self):
        """
        Get the smoothed prediction via majority vote.

        Returns:
            Tuple of (label, avg_confidence):
                - label: The most common prediction in the window (str),
                         or None if buffer is empty.
                - avg_confidence: Average confidence for the majority label.
        """
        if not self.predictions:
            return None, 0.0

        # Count occurrences of each label
        counter = Counter(self.predictions)
        majority_label, count = counter.most_common(1)[0]

        # Calculate average confidence for the majority label
        majority_confidences = [
            conf for pred, conf in zip(self.predictions, self.confidences)
            if pred == majority_label
        ]
        avg_confidence = np.mean(majority_confidences)

        return majority_label, float(avg_confidence)

    def get_stability(self):
        """
        Get the prediction stability (fraction of window agreeing).

        Returns:
            Float between 0 and 1. 1.0 = all predictions agree.
        """
        if not self.predictions:
            return 0.0

        counter = Counter(self.predictions)
        _, count = counter.most_common(1)[0]
        return count / len(self.predictions)

    def reset(self):
        """Clear the prediction buffer."""
        self.predictions.clear()
        self.confidences.clear()

    def is_ready(self):
        """Check if the buffer has enough predictions for a stable vote."""
        return len(self.predictions) >= self.window_size // 2
