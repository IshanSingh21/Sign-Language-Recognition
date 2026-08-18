export class RuleBasedClassifier {
  constructor() {
    this._prevLandmarks = null;
    this._motionBuffer = [];
  }

  _dist(lm, i, j) {
    const dx = lm[i].x - lm[j].x;
    const dy = lm[i].y - lm[j].y;
    const dz = (lm[i].z || 0) - (lm[j].z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  _handScale(lm) {
    return this._dist(lm, 0, 9); // Wrist to Middle MCP
  }

  _isThumbExtended(lm) {
    // thumb tip dist from wrist > thumb MCP dist from wrist * 1.2
    return this._dist(lm, 4, 0) > this._dist(lm, 2, 0) * 1.2;
  }

  _isFingerExtended(lm, mcp, pip, dip, tip) {
    // Assuming y=0 is at the top of the image
    return lm[tip].y < lm[pip].y;
  }

  _isFingerCurled(lm, pip, tip) {
    return lm[tip].y > lm[pip].y;
  }

  classify(landmarks) {
    if (!landmarks || landmarks.length !== 21) {
      return { label: "?", confidence: 0.0 };
    }

    const lm = landmarks;
    const handScale = this._handScale(lm) || 1;

    // Track motion
    let motion = 0;
    if (this._prevLandmarks) {
      let totalDist = 0;
      for (let i = 0; i < 21; i++) {
        const dx = lm[i].x - this._prevLandmarks[i].x;
        const dy = lm[i].y - this._prevLandmarks[i].y;
        const dz = (lm[i].z || 0) - (this._prevLandmarks[i].z || 0);
        totalDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      motion = totalDist / 21;
    }
    
    // Deep copy current landmarks for next frame
    this._prevLandmarks = lm.map(p => ({ x: p.x, y: p.y, z: p.z }));

    this._motionBuffer.push(motion);
    if (this._motionBuffer.length > 15) {
      this._motionBuffer.shift();
    }
    const avgMotion = this._motionBuffer.reduce((a, b) => a + b, 0) / this._motionBuffer.length;

    // Compute finger states
    const thumbExt = this._isThumbExtended(lm);
    const indexExt = this._isFingerExtended(lm, 5, 6, 7, 8);
    const middleExt = this._isFingerExtended(lm, 9, 10, 11, 12);
    const ringExt = this._isFingerExtended(lm, 13, 14, 15, 16);
    const pinkyExt = this._isFingerExtended(lm, 17, 18, 19, 20);

    const numExtended = [thumbExt, indexExt, middleExt, ringExt, pinkyExt].filter(Boolean).length;

    // Compute normalized distances
    const thumbIndexDist = this._dist(lm, 4, 8) / handScale;
    const thumbMiddleDist = this._dist(lm, 4, 12) / handScale;
    const indexMiddleDist = this._dist(lm, 8, 12) / handScale;
    const thumbPinkyDist = this._dist(lm, 4, 20) / handScale;

    // Orientation helper for right vs left hand (assuming right hand if wrist x < pinky MCP x)
    const isRightHand = lm[0].x < lm[17].x;
    const thumbAcrossPalm = isRightHand ? lm[4].x > lm[9].x : lm[4].x < lm[9].x;

    // Rule Evaluations (in priority order)

    // I_LOVE_YOU: thumb + index + pinky extended, middle + ring NOT
    if (thumbExt && indexExt && !middleExt && !ringExt && pinkyExt) {
      return { label: "I_LOVE_YOU", confidence: 0.92 };
    }

    // Y: thumb + pinky extended, others NOT
    if (thumbExt && !indexExt && !middleExt && !ringExt && pinkyExt) {
      return { label: "Y", confidence: 0.90 };
    }

    // L: thumb + index extended, others NOT, thumb_index_dist > 1.2
    if (thumbExt && indexExt && !middleExt && !ringExt && !pinkyExt && thumbIndexDist > 1.2) {
      return { label: "L", confidence: 0.88 };
    }

    // V / U: index + middle extended (NOT thumb)
    if (!thumbExt && indexExt && middleExt && !ringExt && !pinkyExt) {
      if (indexMiddleDist > 0.4) {
        return { label: "V", confidence: 0.90 };
      } else {
        return { label: "U", confidence: 0.85 };
      }
    }

    // W: index + middle + ring extended (NOT thumb, NOT pinky)
    if (!thumbExt && indexExt && middleExt && ringExt && !pinkyExt) {
      return { label: "W", confidence: 0.88 };
    }

    // B (thumb across palm): all five extended, thumb across palm
    if (thumbExt && indexExt && middleExt && ringExt && pinkyExt && thumbAcrossPalm) {
      return { label: "B", confidence: 0.88 };
    }

    // B (normal): all four fingers extended, thumb NOT
    if (!thumbExt && indexExt && middleExt && ringExt && pinkyExt) {
      return { label: "B", confidence: 0.85 };
    }

    // D or Z: index extended, others NOT, thumb_middle_dist < 0.5
    if (!thumbExt && indexExt && !middleExt && !ringExt && !pinkyExt && thumbMiddleDist < 0.5) {
      if (avgMotion > 0.015) {
        return { label: "Z", confidence: 0.75 };
      }
      return { label: "D", confidence: 0.87 };
    }

    // I or J: only pinky extended
    if (!thumbExt && !indexExt && !middleExt && !ringExt && pinkyExt) {
      if (avgMotion > 0.015) {
        return { label: "J", confidence: 0.75 };
      }
      return { label: "I", confidence: 0.88 };
    }

    // R or K: index + middle extended, NOT ring/pinky, index_middle_dist < 0.2
    if (indexExt && middleExt && !ringExt && !pinkyExt && indexMiddleDist < 0.2) {
      if (thumbExt && thumbIndexDist > 0.5) {
        return { label: "K", confidence: 0.80 };
      }
      return { label: "R", confidence: 0.82 };
    }

    // Fist signs
    if (numExtended === 0) {
      // Check E: all fingers curled
      const allCurled = this._isFingerCurled(lm, 6, 8) && 
                        this._isFingerCurled(lm, 10, 12) && 
                        this._isFingerCurled(lm, 14, 16) && 
                        this._isFingerCurled(lm, 18, 20);

      if (lm[4].y < lm[6].y) return { label: "S", confidence: 0.82 }; // thumb tip above index PIP
      if (lm[4].y > lm[5].y && this._dist(lm, 4, 5) / handScale < 0.5) return { label: "T", confidence: 0.80 };
      if (lm[4].y > lm[13].y) return { label: "M", confidence: 0.75 }; // below ring MCP
      if (lm[4].y > lm[9].y) return { label: "N", confidence: 0.75 };  // below middle MCP
      if (allCurled) return { label: "E", confidence: 0.80 };
      
      return { label: "A", confidence: 0.82 }; // default fist
    }

    // Thumb+Index touch (thumb_index_dist < 0.35)
    if (thumbIndexDist < 0.35) {
      if (middleExt && ringExt && pinkyExt) {
        return { label: "F", confidence: 0.88 };
      }
      if (!middleExt && !ringExt && !pinkyExt) {
        return { label: "O", confidence: 0.85 };
      }
    }

    // C: all fingers NOT extended, thumb extended, thumb_index_dist > 0.5
    if (!indexExt && !middleExt && !ringExt && !pinkyExt && thumbExt && thumbIndexDist > 0.5) {
      return { label: "C", confidence: 0.82 };
    }

    // G: thumb + index extended, index pointing sideways
    if (thumbExt && indexExt && !middleExt && !ringExt && !pinkyExt) {
      const dx = Math.abs(lm[8].x - lm[5].x);
      const dy = Math.abs(lm[8].y - lm[5].y);
      if (dx > dy) return { label: "G", confidence: 0.80 };
    }

    // H: index + middle extended, NOT pointing up
    if (!thumbExt && indexExt && middleExt && !ringExt && !pinkyExt) {
      const dx = Math.abs(lm[8].x - lm[5].x);
      const dy = Math.abs(lm[8].y - lm[5].y);
      if (dx > dy) return { label: "H", confidence: 0.80 };
    }

    // X: index hooked
    if (!middleExt && !ringExt && !pinkyExt && lm[7].y < lm[8].y && lm[5].y > lm[6].y) {
      return { label: "X", confidence: 0.78 };
    }

    // Default fallback
    if (indexExt && !middleExt && !ringExt && !pinkyExt) {
      return { label: "D", confidence: 0.70 };
    }

    return { label: "?", confidence: 0.30 };
  }
}
