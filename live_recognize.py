"""
live_recognize.py — Live ASL Sign Language Recognition (No Training Required).

This script uses rule-based classification on MediaPipe hand landmarks to
recognize ASL alphabet letters and common phrases in real-time from your
webcam. No data collection or model training is needed — just run and go!

Controls:
    SPACE      : Add current letter/word to sentence
    BACKSPACE  : Delete last character/word
    C          : Clear sentence
    Q / ESC    : Quit

Supported signs: A-Z (static letters) + I LOVE YOU, YES, NO
Note: J and Z require motion and are detected via basic motion tracking.
"""

import cv2
import numpy as np
import time
from collections import deque

import config
from utils import HandDetector, FeatureExtractor, PredictionSmoother, Visualizer


class RuleBasedClassifier:
    """
    Classifies ASL signs using geometric rules on hand landmarks.

    Uses finger extension states, joint angles, fingertip distances,
    and relative positions to identify static ASL alphabet signs.
    """

    # Landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self):
        self._prev_landmarks = None
        self._motion_buffer = deque(maxlen=15)

    def classify(self, landmarks):
        """
        Classify the hand pose from 21 landmarks.

        Args:
            landmarks: numpy array (21, 3) of [x, y, z] normalized coords.

        Returns:
            (label, confidence) tuple.
        """
        lm = landmarks

        # --- Finger state detection ---
        thumb_ext = self._is_thumb_extended(lm)
        index_ext = self._is_finger_extended(lm, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP)
        middle_ext = self._is_finger_extended(lm, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP)
        ring_ext = self._is_finger_extended(lm, self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP)
        pinky_ext = self._is_finger_extended(lm, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP)

        fingers = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        num_extended = sum(fingers)

        # --- Useful distances (normalized) ---
        scale = self._hand_scale(lm)
        if scale < 1e-6:
            return ("?", 0.0)

        thumb_index_dist = self._dist(lm, self.THUMB_TIP, self.INDEX_TIP) / scale
        thumb_middle_dist = self._dist(lm, self.THUMB_TIP, self.MIDDLE_TIP) / scale
        thumb_ring_dist = self._dist(lm, self.THUMB_TIP, self.RING_TIP) / scale
        thumb_pinky_dist = self._dist(lm, self.THUMB_TIP, self.PINKY_TIP) / scale
        index_middle_dist = self._dist(lm, self.INDEX_TIP, self.MIDDLE_TIP) / scale
        index_ring_dist = self._dist(lm, self.INDEX_TIP, self.RING_TIP) / scale
        middle_ring_dist = self._dist(lm, self.MIDDLE_TIP, self.RING_TIP) / scale
        ring_pinky_dist = self._dist(lm, self.RING_TIP, self.PINKY_TIP) / scale

        # Thumb-to-finger-base distances
        thumb_index_mcp_dist = self._dist(lm, self.THUMB_TIP, self.INDEX_MCP) / scale
        thumb_middle_mcp_dist = self._dist(lm, self.THUMB_TIP, self.MIDDLE_MCP) / scale

        # Finger curl (tip closer to wrist than pip)
        index_curled = self._is_finger_curled(lm, self.INDEX_PIP, self.INDEX_TIP)
        middle_curled = self._is_finger_curled(lm, self.MIDDLE_PIP, self.MIDDLE_TIP)
        ring_curled = self._is_finger_curled(lm, self.RING_PIP, self.RING_TIP)
        pinky_curled = self._is_finger_curled(lm, self.PINKY_PIP, self.PINKY_TIP)

        # Index finger direction (pointing up vs sideways)
        index_pointing_up = lm[self.INDEX_TIP][1] < lm[self.INDEX_MCP][1] - 0.05
        index_pointing_sideways = abs(lm[self.INDEX_TIP][0] - lm[self.INDEX_MCP][0]) > 0.08

        # Thumb position relative to hand
        thumb_across_palm = (
            lm[self.THUMB_TIP][0] > lm[self.INDEX_MCP][0] - 0.02
            if lm[self.WRIST][0] < lm[self.PINKY_MCP][0]  # Right hand
            else lm[self.THUMB_TIP][0] < lm[self.INDEX_MCP][0] + 0.02
        )

        # --- Motion tracking (for J and Z) ---
        motion = 0.0
        if self._prev_landmarks is not None:
            tip_delta = np.linalg.norm(lm[self.INDEX_TIP] - self._prev_landmarks[self.INDEX_TIP])
            pinky_delta = np.linalg.norm(lm[self.PINKY_TIP] - self._prev_landmarks[self.PINKY_TIP])
            motion = max(tip_delta, pinky_delta)
        self._motion_buffer.append(motion)
        self._prev_landmarks = lm.copy()
        avg_motion = np.mean(self._motion_buffer) if self._motion_buffer else 0

        # ═══════════════════════════════════════════════════════
        # CLASSIFICATION RULES
        # ═══════════════════════════════════════════════════════

        # --- I LOVE YOU: Thumb + Index + Pinky extended, Middle + Ring curled ---
        if thumb_ext and index_ext and not middle_ext and not ring_ext and pinky_ext:
            return ("I_LOVE_YOU", 0.92)

        # --- Y: Thumb + Pinky extended, others curled ---
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
            return ("Y", 0.90)

        # --- L: Thumb + Index extended at ~90°, others curled ---
        if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if thumb_index_dist > 1.2:
                return ("L", 0.88)

        # --- V / 2: Index + Middle extended and spread, others curled ---
        if not thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
            if index_middle_dist > 0.4:
                return ("V", 0.90)
            else:
                return ("U", 0.85)

        # --- W / 3: Index + Middle + Ring extended ---
        if not thumb_ext and index_ext and middle_ext and ring_ext and not pinky_ext:
            return ("W", 0.88)

        # --- 4 / B variant: All four fingers extended, thumb curled ---
        if not thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return ("B", 0.85)

        # --- B: All five extended, fingers together ---
        if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            if thumb_across_palm:
                return ("B", 0.88)
            else:
                # Open palm — could be 5 or B
                return ("B", 0.80)

        # --- D: Index extended up, others curled, thumb touches middle ---
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if thumb_middle_dist < 0.5:
                return ("D", 0.87)
            elif thumb_ext:
                return ("L", 0.82)
            else:
                # Just index pointing
                return ("D", 0.78)

        # --- I: Only pinky extended ---
        if not thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
            if avg_motion > 0.015:
                return ("J", 0.75)
            return ("I", 0.88)

        # --- R: Index + Middle crossed ---
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            if index_middle_dist < 0.2:
                return ("R", 0.82)
            # If spread, it's K or V
            if thumb_ext:
                return ("K", 0.80)
            return ("V", 0.78)

        # --- FIST-LIKE signs (no fingers extended) ---
        if num_extended == 0:
            # A: Fist with thumb to the side
            if thumb_across_palm or self._dist(lm, self.THUMB_TIP, self.INDEX_MCP) / scale < 0.6:
                # Thumb wraps or is beside
                # S: Thumb over curled fingers
                if lm[self.THUMB_TIP][1] < lm[self.INDEX_PIP][1]:
                    return ("S", 0.82)
                # T: Thumb between index and middle
                if (lm[self.THUMB_TIP][1] > lm[self.INDEX_MCP][1] and
                    thumb_index_mcp_dist < 0.4):
                    return ("T", 0.80)
                # M: Thumb under 3 fingers
                if (lm[self.THUMB_TIP][1] > lm[self.RING_MCP][1]):
                    return ("M", 0.75)
                # N: Thumb under 2 fingers
                if (lm[self.THUMB_TIP][1] > lm[self.MIDDLE_MCP][1]):
                    return ("N", 0.75)
                return ("A", 0.82)
            # E: All fingertips curled into palm, thumb across
            if (index_curled and middle_curled and ring_curled and pinky_curled):
                return ("E", 0.80)
            return ("S", 0.75)

        # --- Thumb + Index touch (O-like) ---
        if thumb_index_dist < 0.35:
            if middle_ext and ring_ext and pinky_ext:
                # F: Thumb+Index circle, others extended
                return ("F", 0.88)
            elif not middle_ext and not ring_ext and not pinky_ext:
                # O: All fingers curved into circle
                if thumb_middle_dist < 0.5:
                    return ("O", 0.85)
                return ("O", 0.78)
            else:
                return ("F", 0.75)

        # --- C: All fingers curved (cup shape) ---
        if (not index_ext and not middle_ext and not ring_ext and not pinky_ext
                and thumb_ext and thumb_index_dist > 0.5):
            return ("C", 0.82)

        # --- G: Index points sideways, thumb extended ---
        if thumb_ext and index_ext and not middle_ext and index_pointing_sideways:
            return ("G", 0.80)

        # --- H: Index + Middle extended sideways ---
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            if not index_pointing_up:
                return ("H", 0.80)
            return ("U", 0.78)

        # --- P: Index + Middle down, like inverted K ---
        if index_ext and middle_ext and not ring_ext:
            if lm[self.INDEX_TIP][1] > lm[self.INDEX_MCP][1]:
                return ("P", 0.78)

        # --- Q: Thumb + Index pointing down ---
        if thumb_ext and index_ext and not middle_ext:
            if lm[self.INDEX_TIP][1] > lm[self.INDEX_MCP][1]:
                return ("Q", 0.78)

        # --- X: Index hooked (bent at DIP) ---
        if (not middle_ext and not ring_ext and not pinky_ext):
            index_hook = (lm[self.INDEX_DIP][1] < lm[self.INDEX_TIP][1]
                         and lm[self.INDEX_PIP][1] > lm[self.INDEX_MCP][1] + 0.02)
            if index_hook:
                return ("X", 0.78)

        # --- Z: Index extended with motion ---
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if avg_motion > 0.015:
                return ("Z", 0.75)
            return ("D", 0.70)

        # --- Fallback ---
        return ("?", 0.30)

    # ─── Helper Methods ───────────────────────────────────────

    def _is_thumb_extended(self, lm):
        """Check if thumb is extended (tip far from palm center)."""
        thumb_tip_dist = np.linalg.norm(lm[self.THUMB_TIP] - lm[self.WRIST])
        thumb_mcp_dist = np.linalg.norm(lm[self.THUMB_MCP] - lm[self.WRIST])
        return thumb_tip_dist > thumb_mcp_dist * 1.2

    def _is_finger_extended(self, lm, mcp, pip, dip, tip):
        """Check if a finger is extended (tip above PIP joint in y-axis)."""
        return lm[tip][1] < lm[pip][1]

    def _is_finger_curled(self, lm, pip, tip):
        """Check if fingertip is curled (tip below PIP toward wrist)."""
        return lm[tip][1] > lm[pip][1]

    def _dist(self, lm, idx1, idx2):
        """Euclidean distance between two landmarks."""
        return np.linalg.norm(lm[idx1] - lm[idx2])

    def _hand_scale(self, lm):
        """Scale factor: wrist to middle MCP distance."""
        return np.linalg.norm(lm[self.MIDDLE_MCP] - lm[self.WRIST])


def main():
    print("=" * 55)
    print("  ASL SIGN LANGUAGE — LIVE RECOGNITION")
    print("  No training required — works instantly!")
    print("=" * 55)
    print()
    print("Controls:")
    print("  SPACE      : Add letter/word to sentence")
    print("  BACKSPACE  : Delete last character")
    print("  C          : Clear sentence")
    print("  Q / ESC    : Quit")
    print()

    # Initialize components
    detector = HandDetector(
        max_num_hands=config.MP_MAX_HANDS,
        min_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    classifier = RuleBasedClassifier()
    smoother = PredictionSmoother(window_size=config.SMOOTHING_WINDOW)

    # Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sentence = ""
    prev_time = 0
    last_added_time = 0

    print("Webcam opened. Show your hand to start recognizing!\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            #frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # Detect hands
            results = detector.detect(frame)
            frame = detector.draw_landmarks(frame, results)
            landmarks_list = detector.get_landmarks(results)

            if landmarks_list and len(landmarks_list) > 0:
                lm = landmarks_list[0]

                # Classify using rules
                label, confidence = classifier.classify(lm)

                # Smooth prediction
                smoother.add(label, confidence)
                smoothed_label, smoothed_conf = smoother.get_smoothed()
                stability = smoother.get_stability()

                if smoother.is_ready() and smoothed_label != "?":
                    if smoothed_conf >= 0.5:
                        Visualizer.draw_prediction(frame, smoothed_label, smoothed_conf, 10, 100)

                        # Draw stability bar
                        Visualizer.draw_confidence_bar(
                            frame, stability,
                            x=10, y=230, width=200, height=14
                        )
                        # Stability label
                        stab_text = "STABLE" if stability > 0.7 else "UNSTABLE"
                        stab_color = (80, 220, 100) if stability > 0.7 else (0, 200, 255)
                        cv2.putText(frame, stab_text, (220, 243),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stab_color, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f"? (low confidence)", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WARNING, 2, cv2.LINE_AA)
                elif not smoother.is_ready():
                    cv2.putText(frame, "Warming up...", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WARNING, 2, cv2.LINE_AA)
            else:
                smoother.reset()
                Visualizer.draw_no_hand_warning(frame)

            # Draw sentence
            Visualizer.draw_sentence(frame, sentence, 10, 10)

            # Draw FPS
            Visualizer.draw_fps(frame, fps)

            # Draw controls
            Visualizer.draw_controls(frame, mode='recognize')

            # Draw mode indicator
            cv2.putText(frame, "LIVE MODE (Rule-Based)", (w - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

            cv2.imshow("ASL Live Recognition", frame)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                if smoother.is_ready():
                    s_label, s_conf = smoother.get_smoothed()
                    if s_conf >= 0.5 and s_label != "?" and (curr_time - last_added_time) > 0.3:
                        if len(s_label) == 1:
                            sentence += s_label
                        else:
                            if sentence and not sentence.endswith(" "):
                                sentence += " "
                            sentence += s_label.replace("_", " ") + " "
                        last_added_time = curr_time
            elif key == 8:  # BACKSPACE
                if sentence:
                    if sentence[-1] == ' ' and len(sentence) > 1:
                        sentence = sentence.rstrip()
                        while sentence and sentence[-1] != ' ':
                            sentence = sentence[:-1]
                    else:
                        sentence = sentence[:-1]
            elif key == ord('c'):
                sentence = ""

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


if __name__ == '__main__':
    main()
