// app.js - DeafBuddy Pro Major Project Sign Language Translation & Interactive Guide Engine

// 1. Configuration & Global State
const CONFIG = {
    sequenceLength: 30,       // Must match LSTM training sequence length
    numLandmarks: 21,         // Hand landmarks
    coordsPerLandmark: 3,     // X, Y, Z
    totalFeatures: 63,        // 21 * 3 = 63
    confidenceThreshold: 0.50, // Confidence required to lock in a word
    stabilityFrames: 3,       // Consecutive frames required to lock in a word
    cooldownFrames: 12,       // Wait after pushing a word before writing another
    modelPath: 'tfjs_model/model.json?v=' + Date.now(),
    classes: [
        'Bye', 'Closed', 'Eat', 'Good', 'Hello', 'Help', 'How are you',
        'I / Me', 'Love', 'No', 'Open', 'Please', 'Pointer', 'Stop',
        'Thank you', 'Water', 'Welcome', 'What', 'Where', 'Yes', 'You'
    ]
};

// Comprehensive Metadata & 3-Step Physical Tutorial Data for all 21 Signs
const GESTURE_METADATA = {
    'Hello': {
        emoji: '👋',
        category: 'greetings',
        action: 'Wave hand outwards side-to-side',
        step1: 'Position hand at shoulder height with palm facing forward.',
        step2: 'Smoothly sweep wrist X position from left to right.',
        step3: 'Return wrist to center while keeping fingers open.'
    },
    'Bye': {
        emoji: '🖐️',
        category: 'greetings',
        action: 'Wave open palm side-to-side',
        step1: 'Raise flat open palm facing camera.',
        step2: 'Oscillate wrist rapidly left and right.',
        step3: 'Lower hand after 2-3 wave cycles.'
    },
    'Thank you': {
        emoji: '🙏',
        category: 'greetings',
        action: 'Move flat hand down & forward',
        step1: 'Touch fingertips of flat open hand to chin or lips.',
        step2: 'Move hand outward and downward toward camera.',
        step3: 'End with palm facing slightly upward.'
    },
    'Welcome': {
        emoji: '🤝',
        category: 'greetings',
        action: 'Sweep flat hand inward towards chest',
        step1: 'Extend flat open hand outward with palm facing up.',
        step2: 'Sweep arm inward in a curve toward your body.',
        step3: 'Rest hand near waist level.'
    },
    'Please': {
        emoji: '🤲',
        category: 'greetings',
        action: 'Circular motion of flat palm on chest',
        step1: 'Place flat right palm flat against chest.',
        step2: 'Rub hand in clockwise circles.',
        step3: 'Complete 2 small circles.'
    },
    'How are you': {
        emoji: '❓',
        category: 'questions',
        action: 'Push fist forward opening into open palm',
        step1: 'Form loose fist with knuckles together near chest.',
        step2: 'Roll hands outward while extending fingers open.',
        step3: 'Point open palms toward person.'
    },
    'What': {
        emoji: '🤷',
        category: 'questions',
        action: 'Shake open palms side-to-side',
        step1: 'Hold both hands out at waist level, palms up.',
        step2: 'Shake hands horizontally side-to-side.',
        step3: 'Tilt head slightly inquiringly.'
    },
    'Where': {
        emoji: '📍',
        category: 'questions',
        action: 'Shake index finger side-to-side',
        step1: 'Point index finger straight up.',
        step2: 'Pivot wrist side-to-side like a pendulum.',
        step3: 'Hold for 2 seconds.'
    },
    'I / Me': {
        emoji: '👤',
        category: 'questions',
        action: 'Point index finger toward chest',
        step1: 'Extend index finger out.',
        step2: 'Draw finger back to tap chest center.',
        step3: 'Hold briefly at chest.'
    },
    'You': {
        emoji: '👉',
        category: 'questions',
        action: 'Point index finger straight forward',
        step1: 'Point index finger directly at camera/listener.',
        step2: 'Push hand forward 2-3 inches.',
        step3: 'Hold finger stable.'
    },
    'Yes': {
        emoji: '👍',
        category: 'sentiments',
        action: 'Nod closed fist up and down',
        step1: 'Make a fist at shoulder height.',
        step2: 'Bend wrist down and back up (nodding motion).',
        step3: 'Repeat nod twice.'
    },
    'No': {
        emoji: '👎',
        category: 'sentiments',
        action: 'Snap index and middle fingers to thumb',
        step1: 'Extend index and middle fingers with thumb open.',
        step2: 'Snap index & middle fingers down to tap thumb tip.',
        step3: 'Repeat snap twice.'
    },
    'Good': {
        emoji: '👌',
        category: 'sentiments',
        action: 'Touch fingers to mouth and extend out',
        step1: 'Touch flat fingertips or OK-sign to chin.',
        step2: 'Move hand forward into flat open palm.',
        step3: 'Hold palm steady.'
    },
    'Help': {
        emoji: '🆘',
        category: 'sentiments',
        action: 'Lift thumb-up fist with flat palm',
        step1: 'Place thumb-up right fist on flat left palm.',
        step2: 'Lift both hands upward together.',
        step3: 'Hold at chest level.'
    },
    'Love': {
        emoji: '❤️',
        category: 'sentiments',
        action: 'Form ILY sign (Thumb, Index, Pinky extended)',
        step1: 'Extend thumb, index, and pinky (curl middle/ring).',
        step2: 'Move hand forward slightly towards camera.',
        step3: 'Pulse hand gently.'
    },
    'Eat': {
        emoji: '🍽️',
        category: 'actions',
        action: 'Tap pinched fingertips to mouth',
        step1: 'Bring all fingertips to touch thumb tip (bird beak).',
        step2: 'Tap fingertips near lips repeatedly.',
        step3: 'Repeat tap 2-3 times.'
    },
    'Water': {
        emoji: '🚰',
        category: 'actions',
        action: 'Tap W-hand index finger to chin',
        step1: 'Form W shape with index, middle, and ring fingers.',
        step2: 'Tap index finger side against chin twice.',
        step3: 'Lower hand.'
    },
    'Stop': {
        emoji: '🛑',
        category: 'actions',
        action: 'Chop flat hand onto open palm',
        step1: 'Raise flat hand vertical with palm facing inward.',
        step2: 'Bring hand down sharply onto flat open palm.',
        step3: 'Hold abruptly.'
    },
    'Open': {
        emoji: '👐',
        category: 'poses',
        action: 'Hold hand open and flat',
        step1: 'Extend all 5 fingers straight up.',
        step2: 'Hold hand motionless in camera view.',
        step3: 'Keep fingers separated.'
    },
    'Closed': {
        emoji: '✊',
        category: 'poses',
        action: 'Hold a tight closed fist',
        step1: 'Curl all 4 fingers firmly into palm.',
        step2: 'Cross thumb over fingers.',
        step3: 'Hold fist stable.'
    },
    'Pointer': {
        emoji: '👆',
        category: 'poses',
        action: 'Point index finger upward',
        step1: 'Curl thumb, middle, ring, pinky into fist.',
        step2: 'Extend index finger straight upward.',
        step3: 'Hold pointer steady.'
    }
};

// Global Application State
const state = {
    // Inference Engine State
    model: null,
    isModelLoaded: false,
    isCameraActive: false,
    isTranslationPaused: false,
    sequenceBuffer: [],       // Array of 63-element feature arrays
    consecutivePredictions: 0,
    lastPredictedClass: -1,
    activePredictionWord: 'WAITING...',
    currentConfidence: 0,
    cooldownCounter: 0,
    fps: 0,
    lastFrameTime: performance.now(),
    latency: 0,
    noHandFramesCount: 0,
    rawRecognizedWords: [],

    // Practice Mode State
    practice: {
        active: false,
        targetWord: 'Hello',
        streak: 0,
        unlockedCount: 0
    },

    // UI Toolbar State
    activeCategory: 'all',
    searchQuery: '',

    // Dataset Collector State
    isRecording: false,
    recordingLabel: '',
    recordingFrames: [],      
    dataset: []
};

// DOM Elements
const elements = {
    webcamFeed: document.getElementById('webcam-feed'),
    trackingCanvas: document.getElementById('tracking-canvas'),
    statusWebcam: document.getElementById('status-webcam'),
    statusModel: document.getElementById('status-model'),
    statusDetection: document.getElementById('status-detection'),
    hudFps: document.getElementById('hud-fps').querySelector('.value'),
    hudLatency: document.getElementById('hud-latency').querySelector('.value'),
    activeWord: document.getElementById('active-word'),
    confidencePercentage: document.getElementById('confidence-percentage'),
    confidenceBar: document.getElementById('active-confidence-bar'),
    sentenceHistory: document.getElementById('sentence-history'),
    runtimeAcceleration: document.getElementById('runtime-acceleration'),
    bufferText: document.getElementById('telemetry-buffer-text'),
    bufferFill: document.getElementById('telemetry-buffer-fill'),
    memoryStats: document.getElementById('telemetry-memory'),
    btnToggleCamera: document.getElementById('btn-toggle-camera'),
    btnToggleTranslation: document.getElementById('btn-toggle-translation'),
    btnTts: document.getElementById('btn-tts'),
    btnAutoGrammar: document.getElementById('btn-auto-grammar'),
    btnCopyHistory: document.getElementById('btn-copy-history'),
    btnClearHistory: document.getElementById('btn-clear-history'),

    // Guide & Tutorial Elements
    btnOpenTutorial: document.getElementById('btn-open-tutorial'),
    btnOpenPractice: document.getElementById('btn-open-practice'),
    tutorialModal: document.getElementById('tutorial-modal'),
    btnCloseTutorial: document.getElementById('btn-close-tutorial'),
    tutorialCardsContainer: document.getElementById('tutorial-cards-container'),
    inputSearchGestures: document.getElementById('input-search-gestures'),
    gestureReferenceGrid: document.getElementById('gesture-reference-grid'),

    // Practice Challenge Elements
    practiceSection: document.getElementById('practice-section'),
    practiceTargetEmoji: document.getElementById('practice-target-emoji'),
    practiceTargetWord: document.getElementById('practice-target-word'),
    practiceTargetHint: document.getElementById('practice-target-hint'),
    practiceFeedbackText: document.getElementById('practice-feedback-text'),
    practiceStreakBadge: document.getElementById('practice-streak'),
    btnNextChallenge: document.getElementById('btn-next-challenge'),

    // Dataset Builder Elements
    selectRecordClass: document.getElementById('select-record-class'),
    inputCustomClass: document.getElementById('input-custom-class'),
    btnAddClass: document.getElementById('btn-add-class'),
    btnRecordSequence: document.getElementById('btn-record-sequence'),
    recordingProgressWrapper: document.getElementById('recording-progress-wrapper'),
    recordingStateLabel: document.getElementById('recording-state-label'),
    recordingFrameCount: document.getElementById('recording-frame-count'),
    recordingProgressBar: document.getElementById('recording-progress-bar'),
    datasetStatsList: document.getElementById('dataset-stats-list'),
    datasetTotalSamples: document.getElementById('dataset-total-samples'),
    btnDownloadDataset: document.getElementById('btn-download-dataset'),
    btnResetDataset: document.getElementById('btn-reset-dataset')
};

const ctx = elements.trackingCanvas.getContext('2d');
let cameraHelper = null;
let handsDetector = null;

// Application Initialization
window.addEventListener('DOMContentLoaded', async () => {
    setupCanvas();
    syncClassSelector();
    renderGestureGuideCards();
    renderTutorialCards();
    setupEventListeners();
    await loadTensorFlowModel();
    initMediaPipeHands();
});

function setupCanvas() {
    elements.trackingCanvas.width = 640;
    elements.trackingCanvas.height = 480;
    
    ctx.fillStyle = '#04030a';
    ctx.fillRect(0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '16px Space Grotesk';
    ctx.textAlign = 'center';
    ctx.fillText('Camera Inactive. Click "Start Webcam" to begin.', elements.trackingCanvas.width / 2, elements.trackingCanvas.height / 2);
}

function syncClassSelector() {
    elements.selectRecordClass.innerHTML = '';
    const sortedClasses = [...CONFIG.classes].sort();
    sortedClasses.forEach(cls => {
        const meta = GESTURE_METADATA[cls] || { emoji: '🖐️' };
        const opt = document.createElement('option');
        opt.value = cls;
        opt.innerText = `${cls} ${meta.emoji}`;
        elements.selectRecordClass.appendChild(opt);
    });
}

// Render Reference Cards in Grid
function renderGestureGuideCards() {
    elements.gestureReferenceGrid.innerHTML = '';
    const sortedClasses = [...CONFIG.classes].sort();

    sortedClasses.forEach(cls => {
        const meta = GESTURE_METADATA[cls] || { emoji: '🖐️', category: 'all', action: 'Hand sign movement' };
        
        // Category Filter
        if (state.activeCategory !== 'all' && meta.category !== state.activeCategory) return;
        
        // Search Filter
        if (state.searchQuery.trim() !== '') {
            const query = state.searchQuery.toLowerCase();
            if (!cls.toLowerCase().includes(query) && !meta.action.toLowerCase().includes(query)) return;
        }

        const card = document.createElement('div');
        card.className = 'gesture-ref-item';
        card.setAttribute('data-gesture', cls);
        card.innerHTML = `
            <span class="gesture-emoji">${meta.emoji}</span>
            <span class="gesture-name">${cls}</span>
            <span class="gesture-action">${meta.action}</span>
            <button class="btn-card-tutorial" title="Learn Step-by-Step Tutorial" onclick="openSingleTutorial('${cls}')">
                <i class="fa-solid fa-book"></i> Tutorial
            </button>
        `;
        elements.gestureReferenceGrid.appendChild(card);
    });
}

// Render Tutorial Cards in Modal
function renderTutorialCards() {
    elements.tutorialCardsContainer.innerHTML = '';
    const sortedClasses = [...CONFIG.classes].sort();

    sortedClasses.forEach(cls => {
        const meta = GESTURE_METADATA[cls] || {
            emoji: '🖐️', category: 'general', action: 'Gesture motion',
            step1: 'Position hand.', step2: 'Perform gesture motion.', step3: 'Hold position.'
        };

        const item = document.createElement('div');
        item.className = 'tutorial-card glass-panel';
        item.id = `tutorial-card-${cls.replace(/\s+/g, '-')}`;
        item.innerHTML = `
            <div class="tutorial-card-header">
                <span class="tutorial-emoji">${meta.emoji}</span>
                <div class="tutorial-meta">
                    <h3 class="tutorial-word">${cls}</h3>
                    <span class="badge badge-accent category-badge">${meta.category.toUpperCase()}</span>
                </div>
            </div>

            <p class="tutorial-summary">${meta.action}</p>

            <div class="tutorial-steps-timeline">
                <div class="step-item">
                    <span class="step-num">1</span>
                    <div class="step-info">
                        <strong>Starting Position:</strong> ${meta.step1}
                    </div>
                </div>
                <div class="step-item">
                    <span class="step-num">2</span>
                    <div class="step-info">
                        <strong>Movement & Action:</strong> ${meta.step2}
                    </div>
                </div>
                <div class="step-item">
                    <span class="step-num">3</span>
                    <div class="step-info">
                        <strong>Ending Pose:</strong> ${meta.step3}
                    </div>
                </div>
            </div>

            <button class="btn btn-primary btn-try-live" onclick="tryLiveGesture('${cls}')">
                <i class="fa-solid fa-camera"></i> Try It Live
            </button>
        `;
        elements.tutorialCardsContainer.appendChild(item);
    });
}

window.openSingleTutorial = function(cls) {
    elements.tutorialModal.style.display = 'flex';
    const targetCard = document.getElementById(`tutorial-card-${cls.replace(/\s+/g, '-')}`);
    if (targetCard) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        targetCard.classList.add('highlight-tutorial');
        setTimeout(() => targetCard.classList.remove('highlight-tutorial'), 2000);
    }
};

window.tryLiveGesture = function(cls) {
    elements.tutorialModal.style.display = 'none';
    state.practice.targetWord = cls;
    const meta = GESTURE_METADATA[cls] || { emoji: '👋', action: 'Sign movement' };
    elements.practiceTargetEmoji.innerText = meta.emoji;
    elements.practiceTargetWord.innerText = cls;
    elements.practiceTargetHint.innerText = meta.action;
    elements.practiceFeedbackText.innerText = `Ready! Perform '${cls}' in front of camera.`;
    elements.practiceSection.scrollIntoView({ behavior: 'smooth' });

    if (!state.isCameraActive) {
        elements.btnToggleCamera.click();
    }
};

// Event Listeners Setup
function setupEventListeners() {
    elements.btnToggleCamera.addEventListener('click', toggleCamera);
    elements.btnToggleTranslation.addEventListener('click', toggleTranslation);
    elements.btnTts.addEventListener('click', speakSentence);
    elements.btnAutoGrammar.addEventListener('click', autoRefineGrammar);
    elements.btnCopyHistory.addEventListener('click', copyTranscript);
    elements.btnClearHistory.addEventListener('click', clearHistory);

    // Tutorial Modal
    elements.btnOpenTutorial.addEventListener('click', () => elements.tutorialModal.style.display = 'flex');
    elements.btnCloseTutorial.addEventListener('click', () => elements.tutorialModal.style.display = 'none');
    elements.tutorialModal.addEventListener('click', (e) => {
        if (e.target === elements.tutorialModal) elements.tutorialModal.style.display = 'none';
    });

    // Practice Mode Button
    elements.btnOpenPractice.addEventListener('click', () => {
        elements.practiceSection.scrollIntoView({ behavior: 'smooth' });
    });

    // Next Challenge Button
    elements.btnNextChallenge.addEventListener('click', setRandomChallenge);

    // Category Filter Pills
    document.querySelectorAll('.category-pills .pill').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.category-pills .pill').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.activeCategory = e.target.getAttribute('data-category');
            renderGestureGuideCards();
        });
    });

    // Search Bar
    elements.inputSearchGestures.addEventListener('input', (e) => {
        state.searchQuery = e.target.value;
        renderGestureGuideCards();
    });

    // Dataset Builder Actions
    elements.btnAddClass.addEventListener('click', addCustomClass);
    elements.btnRecordSequence.addEventListener('click', startRecordingSequence);
    elements.btnDownloadDataset.addEventListener('click', downloadDataset);
    elements.btnResetDataset.addEventListener('click', resetDataset);
}

// Load TensorFlow.js LayersModel
async function loadTensorFlowModel() {
    try {
        elements.statusModel.setAttribute('data-status', 'loading');
        elements.statusModel.querySelector('.status-text').innerText = 'LOADING...';
        
        state.model = await tf.loadLayersModel(CONFIG.modelPath);
        state.isModelLoaded = true;
        
        elements.statusModel.setAttribute('data-status', 'active');
        elements.statusModel.querySelector('.status-text').innerText = 'READY (21 Classes)';
        elements.runtimeAcceleration.innerText = 'WebGL / WASM';
        
        // Warmup Model
        const dummyInput = tf.zeros([1, CONFIG.sequenceLength, CONFIG.totalFeatures]);
        const warmupPred = state.model.predict(dummyInput);
        warmupPred.dispose();
        dummyInput.dispose();
    } catch (error) {
        console.error('Error loading TensorFlow.js model:', error);
        elements.statusModel.setAttribute('data-status', 'error');
        elements.statusModel.querySelector('.status-text').innerText = 'FAILED';
    }
}

// MediaPipe Hands Initialization
function initMediaPipeHands() {
    handsDetector = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    handsDetector.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.65
    });

    handsDetector.onResults(onHandResults);
}

// Webcam Toggle
async function toggleCamera() {
    if (state.isCameraActive) {
        if (cameraHelper) {
            await cameraHelper.stop();
            cameraHelper = null;
        }
        state.isCameraActive = false;
        elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Webcam';
        elements.btnToggleCamera.classList.remove('btn-danger');
        elements.btnToggleCamera.classList.add('btn-primary');
        elements.btnToggleTranslation.disabled = true;
        elements.statusWebcam.setAttribute('data-status', 'inactive');
        elements.statusWebcam.querySelector('.status-text').innerText = 'OFF';
        setupCanvas();
    } else {
        try {
            cameraHelper = new Camera(elements.webcamFeed, {
                onFrame: async () => {
                    if (state.isCameraActive && handsDetector) {
                        await handsDetector.send({ image: elements.webcamFeed });
                    }
                },
                width: 640,
                height: 480
            });
            await cameraHelper.start();
            state.isCameraActive = true;
            elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Webcam';
            elements.btnToggleCamera.classList.remove('btn-primary');
            elements.btnToggleCamera.classList.add('btn-danger');
            elements.btnToggleTranslation.disabled = false;
            elements.statusWebcam.setAttribute('data-status', 'active');
            elements.statusWebcam.querySelector('.status-text').innerText = 'LIVE';
        } catch (err) {
            console.error('Failed to start camera:', err);
            alert('Camera access denied or unavailable.');
        }
    }
}

function toggleTranslation() {
    state.isTranslationPaused = !state.isTranslationPaused;
    if (state.isTranslationPaused) {
        elements.btnToggleTranslation.innerHTML = '<i class="fa-solid fa-play"></i> Resume Translation';
        elements.btnToggleTranslation.classList.remove('btn-secondary');
        elements.btnToggleTranslation.classList.add('btn-primary');
    } else {
        elements.btnToggleTranslation.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Translation';
        elements.btnToggleTranslation.classList.remove('btn-primary');
        elements.btnToggleTranslation.classList.add('btn-secondary');
    }
}

// MediaPipe Hand Detection & Frame Processing
function onHandResults(results) {
    const now = performance.now();
    state.fps = Math.round(1000 / (now - state.lastFrameTime));
    state.latency = Math.round(now - state.lastFrameTime);
    state.lastFrameTime = now;
    
    elements.hudFps.innerText = state.fps;
    elements.hudLatency.innerText = `${state.latency} ms`;

    ctx.save();
    ctx.clearRect(0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);
    ctx.drawImage(results.image, 0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        elements.statusDetection.setAttribute('data-status', 'active');
        elements.statusDetection.querySelector('.status-text').innerText = 'TRACKING';
        state.noHandFramesCount = 0;

        const landmarks = results.multiHandLandmarks[0];
        drawHandSkeleton(landmarks);

        // Scale-Invariant Wrist & Hand Normalization
        const wrist = landmarks[0];
        const middleMcp = landmarks[9];
        const dx = middleMcp.x - wrist.x;
        const dy = middleMcp.y - wrist.y;
        const dz = middleMcp.z - wrist.z;
        const handScale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;

        const normalizedCoords = [];
        for (let i = 0; i < 21; i++) {
            normalizedCoords.push((landmarks[i].x - wrist.x) / handScale);
            normalizedCoords.push((landmarks[i].y - wrist.y) / handScale);
            normalizedCoords.push((landmarks[i].z - wrist.z) / handScale);
        }

        // Recording mode handler
        if (state.isRecording) {
            handleRecordingFrame(normalizedCoords);
        }

        // Live Inference Handler
        if (!state.isTranslationPaused && state.isModelLoaded) {
            state.sequenceBuffer.push(normalizedCoords);
            if (state.sequenceBuffer.length > CONFIG.sequenceLength) {
                state.sequenceBuffer.shift();
            }

            updateBufferTelemetry();

            if (state.sequenceBuffer.length === CONFIG.sequenceLength) {
                runInference();
            }
        }
    } else {
        elements.statusDetection.setAttribute('data-status', 'none');
        elements.statusDetection.querySelector('.status-text').innerText = 'NONE';
        state.noHandFramesCount++;

        if (state.noHandFramesCount > 15) {
            state.sequenceBuffer = [];
            updateBufferTelemetry();
            state.activePredictionWord = 'WAITING...';
            state.currentConfidence = 0;
            updatePredictionUI();
        }
    }
    ctx.restore();
}

// Draw Hand Skeleton Overlay
function drawHandSkeleton(landmarks) {
    const w = elements.trackingCanvas.width;
    const h = elements.trackingCanvas.height;

    // Draw Bones
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 3;
    HAND_CONNECTIONS.forEach(([i, j]) => {
        const p1 = landmarks[i];
        const p2 = landmarks[j];
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
    });

    // Draw Joints
    landmarks.forEach((pt, idx) => {
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, FINGERTIPS.includes(idx) ? 6 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = FINGERTIPS.includes(idx) ? '#06b6d4' : '#ffffff';
        ctx.fill();
    });
}

// Real-Time Inference Execution
function runInference() {
    if (state.cooldownCounter > 0) {
        state.cooldownCounter--;
        return;
    }

    tf.tidy(() => {
        const inputTensor = tf.tensor3d([state.sequenceBuffer]);
        const prediction = state.model.predict(inputTensor);
        const probabilities = prediction.dataSync();

        let maxIdx = 0;
        let maxProb = probabilities[0];
        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }

        const predictedLabel = CONFIG.classes[maxIdx];
        state.currentConfidence = maxProb;
        state.activePredictionWord = predictedLabel;

        if (maxProb >= CONFIG.confidenceThreshold) {
            if (maxIdx === state.lastPredictedClass) {
                state.consecutivePredictions++;
                if (state.consecutivePredictions >= CONFIG.stabilityFrames) {
                    appendWordToSentence(predictedLabel);
                    checkPracticeModeVerification(predictedLabel);
                    state.consecutivePredictions = 0;
                    state.cooldownCounter = CONFIG.cooldownFrames;
                }
            } else {
                state.lastPredictedClass = maxIdx;
                state.consecutivePredictions = 1;
            }
        } else {
            state.consecutivePredictions = 0;
        }

        updatePredictionUI();
        elements.memoryStats.innerText = `${tf.memory().numTensors} Tensors`;
    });
}

// Update Active Word UI & Guide Highlights
function updatePredictionUI() {
    const refItems = document.querySelectorAll('.gesture-ref-item');
    refItems.forEach(item => item.classList.remove('active-predicted'));

    if (state.activePredictionWord === 'WAITING...') {
        elements.activeWord.innerText = 'WAITING...';
        elements.confidencePercentage.innerText = '0%';
        elements.confidenceBar.style.width = '0%';
    } else {
        elements.activeWord.innerText = state.activePredictionWord.toUpperCase();
        const pct = Math.round(state.currentConfidence * 100);
        elements.confidencePercentage.innerText = `${pct}%`;
        elements.confidenceBar.style.width = `${pct}%`;

        if (state.currentConfidence > 0.82) {
            elements.confidenceBar.style.background = 'linear-gradient(90deg, var(--neon-blue) 0%, var(--neon-cyan) 100%)';
        } else {
            elements.confidenceBar.style.background = 'linear-gradient(90deg, var(--neon-purple) 0%, var(--neon-pink) 100%)';
        }

        const activeWord = state.activePredictionWord.toLowerCase();
        const matchingRef = Array.from(refItems).find(item => item.getAttribute('data-gesture').toLowerCase() === activeWord);
        if (matchingRef) {
            matchingRef.classList.add('active-predicted');
        }
    }
}

// Sentence History & NLP Grammar Engine
function appendWordToSentence(word) {
    // Exclude purely static positioning poses from raw transcript text
    if (['Open', 'Closed', 'Pointer'].includes(word)) return;

    state.rawRecognizedWords.push(word);
    autoRefineGrammar();
}

function autoRefineGrammar() {
    const raw = state.rawRecognizedWords;
    if (raw.length === 0) {
        elements.sentenceHistory.value = '';
        return;
    }

    const text = raw.join(' ');

    // NLP Phrase Replacements for Natural English Sentences
    let refined = text
        .replace(/Hello How are you/gi, "Hello, how are you?")
        .replace(/I \/ Me Love You/gi, "I love you.")
        .replace(/Please Help I \/ Me/gi, "Please help me.")
        .replace(/Please Help/gi, "Please help me.")
        .replace(/What Where/gi, "What is this and where is it?")
        .replace(/Eat Water/gi, "I need food and water.")
        .replace(/Thank you Bye/gi, "Thank you! Goodbye.")
        .replace(/Welcome Good/gi, "You're welcome! Very good.")
        .replace(/Help Water/gi, "Please bring me water!")
        .replace(/How are you Good/gi, "How are you? I am doing good.");

    // Capitalize first letter & ensure trailing punctuation
    refined = refined.charAt(0).toUpperCase() + refined.slice(1);
    if (!/[.!?]$/.test(refined)) refined += '.';

    elements.sentenceHistory.value = refined;
}

// Practice Challenge Verification
function checkPracticeModeVerification(word) {
    if (word.toLowerCase() === state.practice.targetWord.toLowerCase()) {
        state.practice.streak++;
        elements.practiceStreakBadge.innerText = `🔥 Streak: ${state.practice.streak}`;
        elements.practiceFeedbackText.innerHTML = `<span style="color: var(--neon-cyan); font-weight: bold;">🎉 EXCELLENT! Perfect '${word}' sign! +1 Streak!</span>`;
        
        setTimeout(() => {
            setRandomChallenge();
        }, 1800);
    }
}

function setRandomChallenge() {
    const available = CONFIG.classes.filter(c => !['Open', 'Closed', 'Pointer'].includes(c));
    const randomWord = available[Math.floor(Math.random() * available.length)];
    state.practice.targetWord = randomWord;
    
    const meta = GESTURE_METADATA[randomWord] || { emoji: '👋', action: 'Sign movement' };
    elements.practiceTargetEmoji.innerText = meta.emoji;
    elements.practiceTargetWord.innerText = randomWord;
    elements.practiceTargetHint.innerText = meta.action;
    elements.practiceFeedbackText.innerText = `Show the '${randomWord}' gesture in front of the camera...`;
}

// Text-to-Speech
function speakSentence() {
    const text = elements.sentenceHistory.value.trim();
    if (!text) return;
    
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    } else {
        alert('Text-to-speech is not supported in your browser.');
    }
}

function copyTranscript() {
    const text = elements.sentenceHistory.value.trim();
    if (!text) return;
    navigator.clipboard.writeText(text);
    const origText = elements.btnCopyHistory.innerHTML;
    elements.btnCopyHistory.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
    setTimeout(() => elements.btnCopyHistory.innerHTML = origText, 1500);
}

function clearHistory() {
    state.rawRecognizedWords = [];
    elements.sentenceHistory.value = '';
}

function updateBufferTelemetry() {
    const count = state.sequenceBuffer.length;
    elements.bufferText.innerText = `${count} / ${CONFIG.sequenceLength} frames`;
    const fillPct = Math.round((count / CONFIG.sequenceLength) * 100);
    elements.bufferFill.style.width = `${fillPct}%`;
}

// Custom Dataset Builder Functions
function addCustomClass() {
    const newClass = elements.inputCustomClass.value.trim();
    if (!newClass) return;
    
    if (!CONFIG.classes.includes(newClass)) {
        CONFIG.classes.push(newClass);
        GESTURE_METADATA[newClass] = {
            emoji: '✨',
            category: 'custom',
            action: 'Custom recorded gesture',
            step1: 'Position hand.', step2: 'Perform custom gesture.', step3: 'Hold pose.'
        };
        syncClassSelector();
        renderGestureGuideCards();
        renderTutorialCards();
        elements.inputCustomClass.value = '';
    }
}

function startRecordingSequence() {
    if (!state.isCameraActive) {
        alert('Please start the webcam first before recording samples.');
        return;
    }

    state.recordingLabel = elements.selectRecordClass.value;
    state.recordingFrames = [];
    state.isRecording = true;
    
    elements.recordingProgressWrapper.style.display = 'block';
    elements.recordingStateLabel.innerText = `Recording '${state.recordingLabel}'...`;
    elements.recordingFrameCount.innerText = `0 / ${CONFIG.sequenceLength}`;
    elements.recordingProgressBar.style.width = '0%';
    elements.btnRecordSequence.disabled = true;
}

function handleRecordingFrame(coords) {
    state.recordingFrames.push(coords);
    const count = state.recordingFrames.length;
    
    elements.recordingFrameCount.innerText = `${count} / ${CONFIG.sequenceLength}`;
    elements.recordingProgressBar.style.width = `${Math.round((count / CONFIG.sequenceLength) * 100)}%`;

    if (count === CONFIG.sequenceLength) {
        state.dataset.push({
            label: state.recordingLabel,
            sequence: [...state.recordingFrames]
        });

        state.isRecording = false;
        elements.btnRecordSequence.disabled = false;
        elements.recordingStateLabel.innerText = `Saved 1 sample for '${state.recordingLabel}'!`;
        
        updateDatasetStatsUI();

        setTimeout(() => {
            elements.recordingProgressWrapper.style.display = 'none';
        }, 1500);
    }
}

function updateDatasetStatsUI() {
    elements.datasetTotalSamples.innerText = `${state.dataset.length} Samples`;
    elements.btnDownloadDataset.disabled = state.dataset.length === 0;
    elements.btnResetDataset.disabled = state.dataset.length === 0;

    const counts = {};
    state.dataset.forEach(item => {
        counts[item.label] = (counts[item.label] || 0) + 1;
    });

    elements.datasetStatsList.innerHTML = '';
    if (Object.keys(counts).length === 0) {
        elements.datasetStatsList.innerHTML = '<li class="stats-item empty-state">No custom samples recorded yet.</li>';
        return;
    }

    Object.entries(counts).forEach(([lbl, num]) => {
        const li = document.createElement('li');
        li.className = 'stats-item';
        li.innerHTML = `<span>${lbl}</span><strong>${num} samples</strong>`;
        elements.datasetStatsList.appendChild(li);
    });
}

function downloadDataset() {
    if (state.dataset.length === 0) return;
    const jsonStr = JSON.stringify(state.dataset, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dataset.json';
    a.click();
    URL.revokeObjectURL(url);
}

function resetDataset() {
    if (confirm('Are you sure you want to clear all custom recorded dataset samples?')) {
        state.dataset = [];
        updateDatasetStatsUI();
    }
}
