"""
feature_extractor.py — Convert raw hand landmarks into a normalized feature vector.

Extracts 93 features per hand:
  - 63 normalized coordinates (21 landmarks × 3 coords, wrist-centered & scale-normalized)
  - 15 joint angles (MCP, PIP, DIP for each of 5 fingers)
  - 10 fingertip pairwise distances
  -  5 finger extended/curled states
"""

import numpy as np


class FeatureExtractor:
    """
    Transforms raw MediaPipe hand landmarks (21×3) into a classifier-ready
    feature vector of 93 elements.
    """

    # Landmark indices for each finger
    # Each finger: [MCP, PIP, DIP, TIP]
    FINGER_LANDMARKS = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }

    # Fingertip indices
    FINGERTIP_IDS = [4, 8, 12, 16, 20]

    # Wrist index
    WRIST_ID = 0

    # Middle finger MCP (used for scale normalization)
    MIDDLE_MCP_ID = 9

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract(self, landmarks):
        """
        Extract the full 93-element feature vector from raw landmarks.

        Args:
            landmarks: numpy array of shape (21, 3) with [x, y, z] per landmark.

        Returns:
            numpy array of shape (93,) — the feature vector.
            Returns None if landmarks are invalid.
        """
        if landmarks is None or len(landmarks) != 21:
            return None

        landmarks = np.array(landmarks, dtype=np.float64)

        # 1. Normalize coordinates (wrist-centered, scale-invariant)
        norm_coords = self._normalize_coordinates(landmarks)

        # 2. Calculate joint angles
        joint_angles = self._calculate_joint_angles(landmarks)

        # 3. Calculate fingertip pairwise distances
        tip_distances = self._calculate_fingertip_distances(landmarks)

        # 4. Calculate finger states (extended/curled)
        finger_states = self._calculate_finger_states(landmarks)

        # Concatenate all features
        features = np.concatenate([
            norm_coords,      # 63 features
            joint_angles,     # 15 features
            tip_distances,    # 10 features
            finger_states,    #  5 features
        ])

        return features

    def _normalize_coordinates(self, landmarks):
        """
        Normalize landmarks: translate to wrist origin, scale by hand size.

        Args:
            landmarks: (21, 3) array of raw landmarks.

        Returns:
            Flattened array of 63 normalized coordinates.
        """
        # Translate so wrist is at origin
        wrist = landmarks[self.WRIST_ID].copy()
        centered = landmarks - wrist

        # Scale by distance from wrist to middle finger MCP
        scale = np.linalg.norm(centered[self.MIDDLE_MCP_ID])
        if scale > 1e-6:  # Avoid division by zero
            centered = centered / scale

        return centered.flatten()

    def _calculate_joint_angles(self, landmarks):
        """
        Calculate interior angles at each finger joint (MCP, PIP, DIP).

        Uses the dot product formula:
            angle = arccos( (v1 · v2) / (|v1| × |v2|) )

        Args:
            landmarks: (21, 3) array of raw landmarks.

        Returns:
            Array of 15 angles (3 per finger × 5 fingers), in radians
            normalized to [0, 1] range (divided by π).
        """
        angles = []

        for finger_name, ids in self.FINGER_LANDMARKS.items():
            # For thumb: joints are CMC(1), MCP(2), IP(3), TIP(4)
            # For other fingers: MCP, PIP, DIP, TIP
            # We calculate angle at each intermediate joint

            if finger_name == "thumb":
                # Angle at CMC (joint 1): vectors wrist→CMC and CMC→MCP
                joint_points = [
                    (self.WRIST_ID, ids[0], ids[1]),  # Angle at CMC
                    (ids[0], ids[1], ids[2]),          # Angle at MCP
                    (ids[1], ids[2], ids[3]),          # Angle at IP
                ]
            else:
                # Angle at MCP: vectors wrist→MCP and MCP→PIP
                joint_points = [
                    (self.WRIST_ID, ids[0], ids[1]),  # Angle at MCP
                    (ids[0], ids[1], ids[2]),          # Angle at PIP
                    (ids[1], ids[2], ids[3]),          # Angle at DIP
                ]

            for p1_id, joint_id, p2_id in joint_points:
                angle = self._angle_at_joint(
                    landmarks[p1_id],
                    landmarks[joint_id],
                    landmarks[p2_id],
                )
                angles.append(angle / np.pi)  # Normalize to [0, 1]

        return np.array(angles, dtype=np.float64)

    def _angle_at_joint(self, point_a, point_joint, point_b):
        """
        Calculate the angle at `point_joint` formed by vectors
        (point_joint → point_a) and (point_joint → point_b).

        Args:
            point_a: First endpoint (numpy array).
            point_joint: The joint/vertex (numpy array).
            point_b: Second endpoint (numpy array).

        Returns:
            Angle in radians [0, π].
        """
        v1 = point_a - point_joint
        v2 = point_b - point_joint

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors

        return np.arccos(cos_angle)

    def _calculate_fingertip_distances(self, landmarks):
        """
        Calculate pairwise Euclidean distances between all 5 fingertips.

        C(5,2) = 10 pairs: (thumb-index, thumb-middle, ..., ring-pinky).
        Distances are normalized by the hand scale (wrist to middle MCP).

        Args:
            landmarks: (21, 3) array of raw landmarks.

        Returns:
            Array of 10 normalized pairwise distances.
        """
        # Scale factor
        scale = np.linalg.norm(
            landmarks[self.MIDDLE_MCP_ID] - landmarks[self.WRIST_ID]
        )
        if scale < 1e-6:
            scale = 1.0

        tips = landmarks[self.FINGERTIP_IDS]
        distances = []

        for i in range(len(self.FINGERTIP_IDS)):
            for j in range(i + 1, len(self.FINGERTIP_IDS)):
                dist = np.linalg.norm(tips[i] - tips[j]) / scale
                distances.append(dist)

        return np.array(distances, dtype=np.float64)

    def _calculate_finger_states(self, landmarks):
        """
        Determine if each finger is extended (1.0) or curled (0.0).

        Logic:
          - Thumb: Compare tip x-distance from wrist vs MCP x-distance.
          - Other fingers: Tip y should be above (less than) PIP y when extended.

        Args:
            landmarks: (21, 3) array of raw landmarks.

        Returns:
            Array of 5 floats (0.0 or 1.0) for [thumb, index, middle, ring, pinky].
        """
        states = []

        for finger_name, ids in self.FINGER_LANDMARKS.items():
            tip = landmarks[ids[3]]   # Fingertip
            pip = landmarks[ids[1]]   # PIP joint (or MCP for thumb)

            if finger_name == "thumb":
                # Thumb is extended if tip is farther from palm center
                # than the IP joint (in the x-axis direction)
                mcp = landmarks[ids[1]]
                is_extended = (
                    np.linalg.norm(tip - landmarks[self.WRIST_ID])
                    > np.linalg.norm(mcp - landmarks[self.WRIST_ID])
                )
            else:
                # Finger is extended if tip.y < pip.y
                # (in image coords, y decreases going up)
                is_extended = tip[1] < pip[1]

            states.append(1.0 if is_extended else 0.0)

        return np.array(states, dtype=np.float64)

    def get_feature_names(self):
        """
        Get descriptive names for all 93 features (useful for debugging).

        Returns:
            List of 93 feature name strings.
        """
        names = []

        # Normalized coordinates
        for i in range(21):
            for coord in ["x", "y", "z"]:
                names.append(f"lm{i}_{coord}")

        # Joint angles
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            for joint in ["joint1", "joint2", "joint3"]:
                names.append(f"{finger}_{joint}_angle")

        # Fingertip distances
        tip_names = ["thumb", "index", "middle", "ring", "pinky"]
        for i in range(5):
            for j in range(i + 1, 5):
                names.append(f"dist_{tip_names[i]}_{tip_names[j]}")

        # Finger states
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            names.append(f"{finger}_extended")

        return names
