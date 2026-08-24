/**
 * app.js — Main application for ASL Sign Language Recognition.
 *
 * Handles webcam capture, MediaPipe hand landmark detection,
 * ASL classification, prediction smoothing, and UI updates.
 *
 * Uses MediaPipe Tasks Vision (JS) running entirely in the browser.
 */

// Classification is handled by the Flask backend.

// ─── MediaPipe CDN Imports ───────────────────────────────────────
const VISION_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

// ─── DOM Elements ────────────────────────────────────────────────
const loadingOverlay = document.getElementById('loadingOverlay');
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('landmarkCanvas');
const ctx = canvas.getContext('2d');

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const fpsValue = document.getElementById('fpsValue');

const predictionValue = document.getElementById('predictionValue');
const predictionSubtext = document.getElementById('predictionSubtext');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const stabilityStatus = document.getElementById('stabilityStatus');
const stabilityDots = document.getElementById('stabilityDots');

const sentenceText = document.getElementById('sentenceText');
const noHandOverlay = document.getElementById('noHandOverlay');
const webcamBadge = document.getElementById('webcamBadge');

// ─── State ───────────────────────────────────────────────────────
let handLandmarker = null;
let predictionInProgress = false;
let lastPredictionTime = 0;
const PREDICTION_INTERVAL = 1000;
let smoother = new PredictionSmoother(10);

let sentence = "";
let lastAddedTime = 0;
let prevTime = 0;
let isRunning = false;

// ─── Backend Prediction ──────────────────────────────────────────
async function predictWithBackend(landmarks) {
    const landmarkData = landmarks.map(lm => [
        lm.x,
        lm.y,
        lm.z ?? 0
    ]);

    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            landmarks: landmarkData
        })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
        throw new Error(data.error || 'Prediction failed.');
    }

    return {
        label: data.label,
        confidence: Number(data.confidence) || 0
    };
}

// ─── Prediction Smoother ─────────────────────────────────────────
function PredictionSmoother(windowSize) {
    this.windowSize = windowSize;
    this.predictions = [];
    this.confidences = [];
}

PredictionSmoother.prototype.add = function (label, confidence) {
    this.predictions.push(label);
    this.confidences.push(confidence);
    if (this.predictions.length > this.windowSize) {
        this.predictions.shift();
        this.confidences.shift();
    }
};

PredictionSmoother.prototype.getSmoothed = function () {
    if (this.predictions.length === 0) return { label: null, confidence: 0 };

    const counts = {};
    this.predictions.forEach(p => { counts[p] = (counts[p] || 0) + 1; });

    let majorityLabel = null;
    let maxCount = 0;
    for (const [label, count] of Object.entries(counts)) {
        if (count > maxCount) { maxCount = count; majorityLabel = label; }
    }

    const majorityConfs = this.predictions
        .map((p, i) => p === majorityLabel ? this.confidences[i] : null)
        .filter(c => c !== null);
    const avgConf = majorityConfs.reduce((a, b) => a + b, 0) / majorityConfs.length;

    return { label: majorityLabel, confidence: avgConf };
};

PredictionSmoother.prototype.getStability = function () {
    if (this.predictions.length === 0) return 0;
    const counts = {};
    this.predictions.forEach(p => { counts[p] = (counts[p] || 0) + 1; });
    const maxCount = Math.max(...Object.values(counts));
    return maxCount / this.predictions.length;
};

PredictionSmoother.prototype.isReady = function () {
    return this.predictions.length >= Math.floor(this.windowSize / 2);
};

PredictionSmoother.prototype.reset = function () {
    this.predictions = [];
    this.confidences = [];
};

// ─── Label Descriptions ──────────────────────────────────────────
const LABEL_DESCRIPTIONS = {
    'A': 'Fist with thumb at side',
    'B': 'Open hand, fingers together',
    'C': 'Curved hand (cup shape)',
    'D': 'Index up, thumb touches middle',
    'E': 'Fingers curled into palm',
    'F': 'Thumb + index circle, others up',
    'G': 'Index pointing sideways',
    'H': 'Index + middle sideways',
    'I': 'Pinky extended only',
    'J': 'Pinky extended + motion',
    'K': 'Index + middle up, thumb out',
    'L': 'L-shape: thumb + index',
    'M': 'Thumb under three fingers',
    'N': 'Thumb under two fingers',
    'O': 'All fingers form circle',
    'P': 'Index + middle pointing down',
    'Q': 'Thumb + index pointing down',
    'R': 'Index + middle crossed',
    'S': 'Fist with thumb over fingers',
    'T': 'Thumb between index + middle',
    'U': 'Index + middle together up',
    'V': 'Peace sign (spread)',
    'W': 'Three fingers extended',
    'X': 'Index finger hooked',
    'Y': 'Thumb + pinky extended',
    'Z': 'Index traces Z shape',
    'I_LOVE_YOU': 'Thumb + index + pinky',
    '?': 'Unrecognized gesture',
};

// ─── Landmark Drawing ────────────────────────────────────────────
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],// Ring
    [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [5, 9], [9, 13], [13, 17],            // Palm
];

function drawLandmarks(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks || landmarks.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;

    // Draw connections with emerald-gold gradient
    ctx.lineWidth = 3;
    for (const [i, j] of HAND_CONNECTIONS) {
        const a = landmarks[i];
        const b = landmarks[j];

        const gradient = ctx.createLinearGradient(
            a.x * w, a.y * h, b.x * w, b.y * h
        );
        gradient.addColorStop(0, 'rgba(52, 211, 153, 0.85)');
        gradient.addColorStop(1, 'rgba(110, 231, 183, 0.75)');

        ctx.strokeStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(a.x * w, a.y * h);
        ctx.lineTo(b.x * w, b.y * h);
        ctx.stroke();
    }

    // Draw landmarks
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        const x = lm.x * w;
        const y = lm.y * h;

        // Outer glow
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(52, 211, 153, 0.2)';
        ctx.fill();

        // Inner dot — fingertips get gold, joints get emerald
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        const isTip = [4, 8, 12, 16, 20].includes(i);
        ctx.fillStyle = isTip ? '#f5c842' : '#34d399';
        ctx.fill();

        // Cream border
        ctx.strokeStyle = 'rgba(240, 239, 230, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// ─── UI Updates ──────────────────────────────────────────────────
function updatePrediction(label, confidence) {
    const displayLabel = label === 'I_LOVE_YOU' ? 'ILY' : label;
    const description = LABEL_DESCRIPTIONS[label] || '';

    predictionValue.textContent = displayLabel;
    predictionSubtext.textContent = description;

    if (confidence >= 0.8) {
        predictionValue.classList.add('high-confidence');
    } else {
        predictionValue.classList.remove('high-confidence');
    }

    // Update confidence
    const confPct = Math.round(confidence * 100);
    confidenceValue.textContent = confPct + '%';
    confidenceBar.style.width = confPct + '%';

    if (confidence >= 0.85) {
        confidenceBar.className = 'confidence-bar-fill confidence-high';
    } else if (confidence >= 0.6) {
        confidenceBar.className = 'confidence-bar-fill confidence-med';
    } else {
        confidenceBar.className = 'confidence-bar-fill confidence-low';
    }

    // Highlight matching reference card
    document.querySelectorAll('.sign-card').forEach(card => {
        card.classList.remove('active-sign');
        if (card.dataset.sign === label || (label === 'I_LOVE_YOU' && card.dataset.sign === 'ILY')) {
            card.classList.add('active-sign');
        }
    });
}

function updateStability(stability) {
    const dots = stabilityDots.querySelectorAll('.stability-dot');
    const activeDots = Math.round(stability * dots.length);

    dots.forEach((dot, i) => {
        if (i < activeDots) {
            dot.classList.add('active');
            dot.classList.remove('inactive');
        } else {
            dot.classList.remove('active');
            dot.classList.add('inactive');
        }
    });

    if (stability > 0.7) {
        stabilityStatus.textContent = 'STABLE';
        stabilityStatus.className = 'stability-status stable';
    } else if (stability > 0.4) {
        stabilityStatus.textContent = 'ADJUSTING';
        stabilityStatus.className = 'stability-status adjusting';
    } else {
        stabilityStatus.textContent = 'UNSTABLE';
        stabilityStatus.className = 'stability-status unstable';
    }
}

function updateStatus(state) {
    statusDot.className = 'status-dot ' + state;
    const messages = {
        'detecting': 'Hand Detected',
        'warming': 'Warming Up...',
        'nohand': 'No Hand Detected',
        'loading': 'Loading...',
    };
    statusText.textContent = messages[state] || state;
}

function updateSentence() {
    sentenceText.textContent = sentence || '';
}

// ─── Main Detection Loop ─────────────────────────────────────────
function detectFrame() {
    if (!isRunning || !handLandmarker) return;

    const now = performance.now();

    // FPS
    if (prevTime > 0) {
        const fps = 1000 / (now - prevTime);
        fpsValue.textContent = Math.round(fps);
    }
    prevTime = now;

    // Run detection
    const results = handLandmarker.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];

        // Draw landmarks on canvas
        drawLandmarks(landmarks);

        // Send landmarks to Python/Flask SVM backend
const predictionNow = performance.now();

if (
    !predictionInProgress &&
    predictionNow - lastPredictionTime >= PREDICTION_INTERVAL
) {
    predictionInProgress = true;
    lastPredictionTime = predictionNow;

    predictWithBackend(landmarks)
        .then(({ label, confidence }) => {
            smoother.add(label, confidence);

            const smoothed = smoother.getSmoothed();
            const stability = smoother.getStability();

            if (smoother.isReady() && smoothed.label !== '?') {
                updateStatus('detecting');
                updatePrediction(smoothed.label, smoothed.confidence);
                updateStability(stability);
            } else {
                updateStatus('warming');
                updateStability(0);
            }
        })
        .catch(error => {
            console.error('Backend prediction error:', error);
        })
        .finally(() => {
            predictionInProgress = false;
        });
}

     noHandOverlay.classList.remove('visible');
     
    } else {
        // No hand
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        smoother.reset();
        noHandOverlay.classList.add('visible');
        updateStatus('nohand');
        updateStability(0);
        predictionValue.textContent = '—';
        predictionSubtext.textContent = 'Waiting for hand...';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '0%';
    }

    requestAnimationFrame(detectFrame);
}

// ─── Keyboard Handling ───────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    const now = Date.now();

    if (e.code === 'Space') {
        e.preventDefault();
        if (smoother.isReady()) {
            const { label, confidence } = smoother.getSmoothed();
            if (confidence >= 0.5 && label !== '?' && (now - lastAddedTime) > 300) {
                if (label.length === 1) {
                    sentence += label;
                } else {
                    if (sentence && !sentence.endsWith(' ')) sentence += ' ';
                    sentence += label.replace(/_/g, ' ') + ' ';
                }
                lastAddedTime = now;
                updateSentence();
            }
        }
    } else if (e.code === 'Backspace') {
        e.preventDefault();
        if (sentence.length > 0) {
            if (sentence.endsWith(' ') && sentence.length > 1) {
                sentence = sentence.trimEnd();
                while (sentence.length > 0 && !sentence.endsWith(' ')) {
                    sentence = sentence.slice(0, -1);
                }
            } else {
                sentence = sentence.slice(0, -1);
            }
            updateSentence();
        }
    } else if (e.key === 'c' || e.key === 'C') {
        // Only clear if not typing in an input
        if (document.activeElement === document.body) {
            sentence = '';
            updateSentence();
        }
    } else if (e.code === 'Escape') {
        isRunning = false;
        updateStatus('stopped');
    }
});

// ─── Initialization ──────────────────────────────────────────────
async function startCamera() {
    // Try multiple camera configurations
    const configs = [
        { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } },
        { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } },
        { video: true },
    ];

    for (const config of configs) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(config);
            video.srcObject = stream;
            await video.play();
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log(`Webcam started: ${video.videoWidth}x${video.videoHeight}`);
            return true;
        } catch (err) {
            console.warn('Camera config failed:', config, err.message);
        }
    }
    return false;
}

function showError(title, message) {
    const overlay = loadingOverlay;
    overlay.classList.remove('hidden');
    overlay.querySelector('.spinner').style.display = 'none';
    overlay.querySelector('.loading-title').textContent = title;
    overlay.querySelector('.loading-subtitle').textContent = message;

    // Add retry button if not already present
    if (!document.getElementById('retryBtn')) {
        const btn = document.createElement('button');
        btn.id = 'retryBtn';
        btn.textContent = '🔄 Retry';
        btn.style.cssText = `
            margin-top: 24px; padding: 12px 36px; border: none; border-radius: 12px;
            background: linear-gradient(135deg, #6c5ce7, #00cec9); color: #fff;
            font-size: 16px; font-weight: 600; cursor: pointer; font-family: 'Inter', sans-serif;
            transition: transform 0.2s, box-shadow 0.2s;
        `;
        btn.onmouseenter = () => { btn.style.transform = 'scale(1.05)'; btn.style.boxShadow = '0 0 20px rgba(108,92,231,0.5)'; };
        btn.onmouseleave = () => { btn.style.transform = 'scale(1)'; btn.style.boxShadow = 'none'; };
        btn.onclick = () => {
            btn.remove();
            overlay.querySelector('.spinner').style.display = '';
            overlay.querySelector('.loading-title').textContent = 'Retrying...';
            overlay.querySelector('.loading-subtitle').textContent = 'Connecting to camera...';
            init();
        };
        overlay.querySelector('.loading-content').appendChild(btn);
    }
}

async function init() {
    try {
        // 1. Load MediaPipe Vision
        updateStatus('loading');

        if (!handLandmarker) {
            const { FilesetResolver, HandLandmarker } = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest'
            );

            const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

            // Try GPU first, fall back to CPU
            try {
                handLandmarker = await HandLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
                    runningMode: 'VIDEO',
                    numHands: 1,
                    minHandDetectionConfidence: 0.7,
                    minHandPresenceConfidence: 0.6,
                    minTrackingConfidence: 0.6,
                });
            } catch {
                console.warn('GPU delegate failed, falling back to CPU');
                handLandmarker = await HandLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
                    runningMode: 'VIDEO',
                    numHands: 1,
                    minHandDetectionConfidence: 0.7,
                    minHandPresenceConfidence: 0.6,
                    minTrackingConfidence: 0.6,
                });
            }
            console.log('MediaPipe HandLandmarker loaded successfully.');
        }

        // 2. Start webcam
        loadingOverlay.querySelector('.loading-subtitle').textContent = 'Starting camera...';
        const cameraOk = await startCamera();

        if (!cameraOk) {
            showError('Camera Error',
                'Could not access camera. Close any other apps using the camera (like other browser tabs, Zoom, Teams, etc.), then click Retry.');
            return;
        }

        // 3. Hide loading, start detection
        loadingOverlay.classList.add('hidden');
        isRunning = true;
        updateStatus('nohand');

        requestAnimationFrame(detectFrame);

    } catch (err) {
        console.error('Initialization error:', err);
        showError('Error', err.message || 'Failed to initialize. Check camera permissions.');
    }
}

// Start!
init();

