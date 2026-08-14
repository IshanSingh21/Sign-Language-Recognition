"""
visualization.py — OpenCV drawing utilities for the gesture recognition UI.

Provides functions for drawing panels, text overlays, confidence bars,
recording indicators, and the sentence display on webcam frames.
"""

import cv2
import numpy as np


class Visualizer:
    """Drawing utilities for the gesture recognition UI overlay."""

    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    def __init__(self):
        """Initialize the visualizer."""
        pass

    @staticmethod
    def draw_rounded_rect(frame, pt1, pt2, color, radius=15, thickness=-1, alpha=0.7):
        """
        Draw a rounded rectangle with optional transparency.

        Args:
            frame: Image to draw on.
            pt1: Top-left corner (x, y).
            pt2: Bottom-right corner (x, y).
            color: BGR color tuple.
            radius: Corner radius.
            thickness: -1 for filled, >0 for border only.
            alpha: Transparency (0=transparent, 1=opaque).
        """
        overlay = frame.copy()
        x1, y1 = pt1
        x2, y2 = pt2

        # Clamp radius
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

        if thickness == -1:
            # Filled rounded rectangle
            # Main body
            cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Corners
            cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Border only
            cv2.line(overlay, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(overlay, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(overlay, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(overlay, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def draw_prediction(frame, label, confidence, x=20, y=80):
        """
        Draw the current prediction with large text and confidence.

        Args:
            frame: Image to draw on.
            label: Predicted gesture label.
            confidence: Confidence value (0-1).
            x, y: Position for the text.
        """
        h, w = frame.shape[:2]

        # Background panel
        panel_w = 350
        panel_h = 120
        Visualizer.draw_rounded_rect(
            frame, (x, y - 10), (x + panel_w, y + panel_h),
            (30, 30, 40), radius=12, alpha=0.85,
        )

        # Label "PREDICTION"
        cv2.putText(
            frame, "PREDICTION",
            (x + 15, y + 22),
            Visualizer.FONT, 0.5, (150, 150, 160), 1, cv2.LINE_AA,
        )

        # Main prediction text
        display_label = label.replace("_", " ") if label else "---"
        font_scale = 1.8 if len(display_label) <= 3 else 1.2
        cv2.putText(
            frame, display_label,
            (x + 15, y + 75),
            Visualizer.FONT_BOLD, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Confidence percentage
        conf_text = f"{confidence * 100:.0f}%"
        if confidence >= 0.85:
            conf_color = (80, 220, 100)   # Green
        elif confidence >= 0.60:
            conf_color = (0, 200, 255)    # Yellow
        else:
            conf_color = (60, 60, 255)    # Red

        cv2.putText(
            frame, conf_text,
            (x + panel_w - 90, y + 75),
            Visualizer.FONT_BOLD, 1.2, conf_color, 2, cv2.LINE_AA,
        )

        # Confidence bar
        Visualizer.draw_confidence_bar(
            frame, confidence,
            x=x + 15, y=y + 95,
            width=panel_w - 30, height=12,
        )

    @staticmethod
    def draw_confidence_bar(frame, confidence, x, y, width=200, height=10):
        """
        Draw a horizontal confidence bar with gradient coloring.

        Args:
            frame: Image to draw on.
            confidence: Value between 0 and 1.
            x, y: Top-left position.
            width: Bar width in pixels.
            height: Bar height in pixels.
        """
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 70), -1)

        # Filled portion
        fill_width = int(width * min(confidence, 1.0))
        if fill_width > 0:
            if confidence >= 0.85:
                color = (80, 220, 100)
            elif confidence >= 0.60:
                color = (0, 200, 255)
            else:
                color = (60, 60, 255)

            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 110), 1)

    @staticmethod
    def draw_sentence(frame, sentence, x=20, y=None):
        """
        Draw the accumulated sentence at the top of the frame.

        Args:
            frame: Image to draw on.
            sentence: The sentence string to display.
            x: Left margin.
            y: Top position (defaults to top of frame).
        """
        h, w = frame.shape[:2]
        if y is None:
            y = 10

        panel_w = w - 40
        panel_h = 55

        Visualizer.draw_rounded_rect(
            frame, (x, y), (x + panel_w, y + panel_h),
            (25, 25, 35), radius=10, alpha=0.9,
        )

        # Label
        cv2.putText(
            frame, "SENTENCE:",
            (x + 12, y + 18),
            Visualizer.FONT, 0.4, (140, 140, 150), 1, cv2.LINE_AA,
        )

        # Sentence text (truncate if too long)
        display_text = sentence if len(sentence) <= 50 else "..." + sentence[-47:]
        cv2.putText(
            frame, display_text,
            (x + 12, y + 42),
            Visualizer.FONT_BOLD, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def draw_recording_indicator(frame, is_recording, label, count):
        """
        Draw the recording status indicator.

        Args:
            frame: Image to draw on.
            is_recording: Whether actively capturing.
            label: Current target label.
            count: Number of samples collected for this label.
        """
        h, w = frame.shape[:2]

        x = w - 320
        y = 10
        panel_w = 300
        panel_h = 80

        border_color = (0, 0, 255) if is_recording else (100, 100, 110)

        Visualizer.draw_rounded_rect(
            frame, (x, y), (x + panel_w, y + panel_h),
            (25, 25, 35), radius=10, alpha=0.9,
        )

        if is_recording:
            # Red border when recording
            cv2.rectangle(
                frame, (x, y), (x + panel_w, y + panel_h),
                (0, 0, 255), 2,
            )
            # Blinking red dot
            cv2.circle(frame, (x + 20, y + 25), 8, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC",
                (x + 35, y + 30),
                Visualizer.FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "READY",
                (x + 15, y + 30),
                Visualizer.FONT, 0.5, (80, 220, 100), 1, cv2.LINE_AA,
            )

        # Current label
        display_label = label.replace("_", " ") if label else "---"
        cv2.putText(
            frame, f"Label: {display_label}",
            (x + 15, y + 55),
            Visualizer.FONT_BOLD, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Sample count
        cv2.putText(
            frame, f"n={count}",
            (x + panel_w - 80, y + 55),
            Visualizer.FONT, 0.6, (200, 200, 210), 1, cv2.LINE_AA,
        )

    @staticmethod
    def draw_label_selector(frame, labels, current_idx, counts, y_start=220):
        """
        Draw a scrollable label selector panel.

        Args:
            frame: Image to draw on.
            labels: List of all label strings.
            current_idx: Index of currently selected label.
            counts: Dict mapping label → sample count.
            y_start: Y position to start drawing.
        """
        h, w = frame.shape[:2]
        x = 20
        panel_w = 350

        # Show a window of labels around the current one
        visible = 8
        start = max(0, current_idx - visible // 2)
        end = min(len(labels), start + visible)
        start = max(0, end - visible)

        panel_h = (end - start) * 28 + 30

        Visualizer.draw_rounded_rect(
            frame, (x, y_start), (x + panel_w, y_start + panel_h),
            (25, 25, 35), radius=10, alpha=0.85,
        )

        cv2.putText(
            frame, f"LABELS ({current_idx + 1}/{len(labels)})",
            (x + 12, y_start + 20),
            Visualizer.FONT, 0.45, (140, 140, 150), 1, cv2.LINE_AA,
        )

        for i in range(start, end):
            row_y = y_start + 40 + (i - start) * 28
            label = labels[i]
            count = counts.get(label, 0)

            if i == current_idx:
                # Highlight selected
                cv2.rectangle(
                    frame,
                    (x + 8, row_y - 14),
                    (x + panel_w - 8, row_y + 10),
                    (60, 60, 80), -1,
                )
                text_color = (255, 200, 80)
                marker = ">"
            else:
                text_color = (200, 200, 210)
                marker = " "

            # Count indicator
            if count >= 500:
                status_color = (80, 220, 100)   # Green — target met
            elif count >= 200:
                status_color = (0, 200, 255)    # Yellow — minimum met
            else:
                status_color = (100, 100, 120)  # Gray — needs more

            cv2.circle(frame, (x + 20, row_y - 2), 4, status_color, -1)

            display = label.replace("_", " ")
            cv2.putText(
                frame, f"{marker} {display}",
                (x + 32, row_y),
                Visualizer.FONT, 0.45, text_color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{count}",
                (x + panel_w - 60, row_y),
                Visualizer.FONT, 0.4, status_color, 1, cv2.LINE_AA,
            )

    @staticmethod
    def draw_fps(frame, fps):
        """
        Draw FPS counter in the corner.

        Args:
            frame: Image to draw on.
            fps: Frames per second value.
        """
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.0f}"
        cv2.putText(
            frame, text,
            (w - 130, h - 15),
            Visualizer.FONT, 0.5, (140, 140, 150), 1, cv2.LINE_AA,
        )

    @staticmethod
    def draw_controls(frame, mode="collect"):
        """
        Draw control hints at the bottom of the frame.

        Args:
            frame: Image to draw on.
            mode: 'collect' or 'recognize'.
        """
        h, w = frame.shape[:2]
        y = h - 50

        panel_h = 40
        Visualizer.draw_rounded_rect(
            frame, (10, y), (w - 10, y + panel_h),
            (20, 20, 30), radius=8, alpha=0.85,
        )

        if mode == "collect":
            controls = "SPACE: Auto-capture | S: Single capture | N/P: Next/Prev label | Q: Quit"
        else:
            controls = "SPACE: Add to sentence | BACKSPACE: Delete | C: Clear | Q: Quit"

        cv2.putText(
            frame, controls,
            (25, y + 26),
            Visualizer.FONT, 0.4, (180, 180, 190), 1, cv2.LINE_AA,
        )

    @staticmethod
    def draw_no_hand_warning(frame):
        """
        Draw a 'No hand detected' warning message.

        Args:
            frame: Image to draw on.
        """
        h, w = frame.shape[:2]
        text = "No hand detected — show your hand to the camera"
        text_size = cv2.getTextSize(text, Visualizer.FONT, 0.6, 1)[0]
        x = (w - text_size[0]) // 2
        y = h // 2

        Visualizer.draw_rounded_rect(
            frame, (x - 15, y - 25), (x + text_size[0] + 15, y + 10),
            (30, 30, 60), radius=8, alpha=0.8,
        )

        cv2.putText(
            frame, text,
            (x, y),
            Visualizer.FONT, 0.6, (100, 100, 255), 1, cv2.LINE_AA,
        )
