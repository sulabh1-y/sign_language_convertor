// app.js - Sign Language Translation & Dataset Collector Logic

// 1. Configuration & Global State
const CONFIG = {
    sequenceLength: 30,       // Must match LSTM training sequence length
    numLandmarks: 21,         // Hand landmarks
    coordsPerLandmark: 3,     // X, Y, Z
    totalFeatures: 63,        // 21 * 3 = 63
    confidenceThreshold: 0.82, // Confidence required to trigger stabilization
    stabilityFrames: 7,       // Consecutive frames required to lock in a word
    cooldownFrames: 25,       // Wait after pushing a word before writing another
    modelPath: 'tfjs_model/model.json',
    classes: ['Closed', 'Open', 'Pointer']
};

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

    // Dataset Collector State
    isRecording: false,
    recordingLabel: '',
    recordingFrames: [],      // Array of 63-element frames for active record
    dataset: [],              // Array of { label: string, sequence: number[][] }
    countdownValue: 0,        // Active countdown overlay (3, 2, 1)
    countdownInterval: null
};

// Hand Connections Map (Pairs of joint IDs)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index Finger
    [5, 9], [9, 10], [10, 11], [11, 12],  // Middle Finger
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring Finger
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
];

// Fingertips indices
const FINGERTIPS = [4, 8, 12, 16, 20];

// 2. DOM Elements
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
    btnClearHistory: document.getElementById('btn-clear-history'),

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

// 3. Application Initialization
window.addEventListener('DOMContentLoaded', async () => {
    setupCanvas();
    syncClassSelector();
    setupEventListeners();
    await loadTensorFlowModel();
    initMediaPipeHands();
});

// Set canvas dimensions and draw standby message
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

// Populate class selector dropdown based on CONFIG.classes
function syncClassSelector() {
    // Sort classes alphabetically to ensure perfect index alignment with Python's model output mapping!
    CONFIG.classes.sort();
    
    elements.selectRecordClass.innerHTML = '';
    CONFIG.classes.forEach(cls => {
        const opt = document.createElement('option');
        opt.value = cls;
        opt.innerText = cls;
        elements.selectRecordClass.appendChild(opt);
    });
    
    // Update labels count
    document.getElementById('telemetry-classes').innerText = `${CONFIG.classes.length} (${CONFIG.classes.join(', ')})`;
}

// Helper to dynamically resolve TensorFlow.js graph signature discrepancies
async function executeModel(model, inputTensor) {
    const trials = [
        { name: "Dict 'keras_tensor'", fn: async () => await model.executeAsync({ 'keras_tensor': inputTensor }) },
        { name: "Dict 'keras_tensor' with outputs ['Identity']", fn: async () => await model.executeAsync({ 'keras_tensor': inputTensor }, ['Identity']) },
        { name: "Dict 'keras_tensor' with outputs ['Identity:0']", fn: async () => await model.executeAsync({ 'keras_tensor': inputTensor }, ['Identity:0']) },
        { name: "Dict 'keras_tensor:0'", fn: async () => await model.executeAsync({ 'keras_tensor:0': inputTensor }) },
        { name: "Dict 'keras_tensor:0' with outputs ['Identity:0']", fn: async () => await model.executeAsync({ 'keras_tensor:0': inputTensor }, ['Identity:0']) },
        { name: "Raw tensor", fn: async () => await model.executeAsync(inputTensor) }
    ];
    let errors = [];
    for (const trial of trials) {
        try {
            return await trial.fn();
        } catch (err) {
            errors.push(`[${trial.name}]: ${err.message}`);
        }
    }
    throw new Error("All trials failed:\n" + errors.join("\n"));
}

// 4. Load TensorFlow.js Model
async function loadTensorFlowModel() {
    try {
        console.log("Loading TensorFlow.js model from:", CONFIG.modelPath);
        elements.statusModel.setAttribute('data-status', 'loading');
        elements.statusModel.querySelector('.status-text').innerText = 'LOADING';

        state.model = await tf.loadGraphModel(CONFIG.modelPath);
        state.isModelLoaded = true;
        
        const backend = tf.getBackend().toUpperCase();
        console.log(`Model loaded successfully. Acceleration: ${backend}`);
        
        elements.statusModel.setAttribute('data-status', 'ready');
        elements.statusModel.querySelector('.status-text').innerText = 'READY';
        elements.runtimeAcceleration.innerText = backend;
        
        // Warm up model using executeAsync to handle dynamic control flow ops (LSTM Exit ops)
        const dummyInput = tf.zeros([1, CONFIG.sequenceLength, CONFIG.totalFeatures]);
        const dummyOutput = await executeModel(state.model, dummyInput);
        dummyInput.dispose();
        if (Array.isArray(dummyOutput)) {
            dummyOutput.forEach(t => t.dispose());
        } else {
            dummyOutput.dispose();
        }
        
        updateMemoryDiagnostics();
        elements.btnToggleTranslation.removeAttribute('disabled');
    } catch (error) {
        elements.statusModel.setAttribute('data-status', 'inactive');
        elements.statusModel.querySelector('.status-text').innerText = 'ERR: ' + error.message.substring(0, 30);
        
        // Output detailed error to history block
        if (elements.sentenceHistory) {
            const inputsStr = state.model && state.model.inputs ? JSON.stringify(state.model.inputs) : 'undefined';
            const outputsStr = state.model && state.model.outputs ? JSON.stringify(state.model.outputs) : 'undefined';
            elements.sentenceHistory.innerText = `[LOAD ERROR] ${error.message}\n\nModel Inputs: ${inputsStr}\nModel Outputs: ${outputsStr}\n\nStack:\n${error.stack || ''}`;
        }
        
        // Trigger browser alert popup so the user can see it immediately
        alert("TFJS LOAD ERROR:\n" + error.message);
        
        createMockModel();
    }
}

// Fallback Mock Model for UI robustness if files aren't converted yet
function createMockModel() {
    state.model = {
        predict: (tensor) => {
            return tf.tidy(() => {
                const batch = tensor.shape[0];
                const logits = tf.randomNormal([batch, CONFIG.classes.length]);
                return tf.softmax(logits);
            });
        },
        executeAsync: async (inputs) => {
            return tf.tidy(() => {
                let batch = 1;
                const tensor = inputs instanceof tf.Tensor ? inputs : Object.values(inputs)[0];
                if (tensor && tensor.shape) {
                    batch = tensor.shape[0];
                }
                const logits = tf.randomNormal([batch, CONFIG.classes.length]);
                return tf.softmax(logits);
            });
        }
    };
    state.isModelLoaded = true;
    elements.statusModel.setAttribute('data-status', 'ready');
    elements.statusModel.querySelector('.status-text').innerText = 'FALLBACK';
    elements.runtimeAcceleration.innerText = 'CPU (MOCK)';
    elements.btnToggleTranslation.removeAttribute('disabled');
}

// 5. Initialize MediaPipe Hands
function initMediaPipeHands() {
    console.log("Initializing MediaPipe Hands...");
    
    handsDetector = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    handsDetector.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    handsDetector.onResults(onResults);
}

// 6. MediaPipe Frame Callback
function onResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);
    
    // Draw background video feed
    if (results.image) {
        ctx.drawImage(results.image, 0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);
    } else {
        ctx.fillStyle = '#04030a';
        ctx.fillRect(0, 0, elements.trackingCanvas.width, elements.trackingCanvas.height);
    }
    
    // Calculate FPS
    const now = performance.now();
    state.fps = Math.round(1000 / (now - state.lastFrameTime));
    state.lastFrameTime = now;
    elements.hudFps.innerText = state.fps.toString().padStart(2, '0');

    // Decrease active prediction cooldown counter
    if (state.cooldownCounter > 0) {
        state.cooldownCounter--;
    }

    // Process Hand Landmarks
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        state.noHandFramesCount = 0;
        elements.statusDetection.setAttribute('data-status', 'detected');
        elements.statusDetection.querySelector('.status-text').innerText = 'DETECTED';
        
        const handLandmarks = results.multiHandLandmarks[0];
        
        // Extract coordinates with wrist-relative normalization and max-absolute scaling
        const base_x = handLandmarks[0].x;
        const base_y = handLandmarks[0].y;
        
        const relativeCoords = [];
        for (let i = 0; i < handLandmarks.length; i++) {
            relativeCoords.push(handLandmarks[i].x - base_x);
            relativeCoords.push(handLandmarks[i].y - base_y);
        }
        
        let maxVal = 0;
        for (let i = 0; i < relativeCoords.length; i++) {
            const absVal = Math.abs(relativeCoords[i]);
            if (absVal > maxVal) {
                maxVal = absVal;
            }
        }
        
        const coords = [];
        for (let i = 0; i < handLandmarks.length; i++) {
            const normX = maxVal > 0 ? (handLandmarks[i].x - base_x) / maxVal : 0;
            const normY = maxVal > 0 ? (handLandmarks[i].y - base_y) / maxVal : 0;
            coords.push(normX, normY, 0.0); // Pad Z with 0.0 to match the 3D shape (63 features)
        }
        
        // 1. Handle Active Dataset Recording
        if (state.isRecording && state.countdownValue === 0) {
            handleDatasetRecording(coords);
        }
        
        // 2. Manage Inference Sequence Buffer (only run inference if not recording)
        if (!state.isRecording) {
            state.sequenceBuffer.push(coords);
            if (state.sequenceBuffer.length > CONFIG.sequenceLength) {
                state.sequenceBuffer.shift();
            }
            updateBufferTelemetry();
            
            // Run inference
            if (state.isModelLoaded && !state.isTranslationPaused) {
                if (state.sequenceBuffer.length === CONFIG.sequenceLength) {
                    runInference();
                }
            }
        }
        
        // 3. Draw visual hand skeleton overlay
        drawSkeleton(handLandmarks);
    } else {
        // No hand visible
        state.noHandFramesCount++;
        
        if (state.isRecording && state.countdownValue === 0) {
            elements.recordingStateLabel.innerText = `Recording PAUSED: Show hand!`;
        }
        
        if (state.noHandFramesCount > 5) {
            elements.statusDetection.setAttribute('data-status', 'none');
            elements.statusDetection.querySelector('.status-text').innerText = 'NONE';
            
            // Clear prediction states if hand disappears
            if (!state.isRecording) {
                state.activePredictionWord = 'WAITING...';
                state.currentConfidence = 0;
                updatePredictionUI();
                
                // Decay rolling inference buffer
                if (state.sequenceBuffer.length > 0) {
                    state.sequenceBuffer.shift();
                    updateBufferTelemetry();
                }
            }
        }
    }
    
    // Draw graphical countdown overlay if countdown is active
    if (state.countdownValue > 0) {
        drawCountdownOverlay();
    }
    
    ctx.restore();
    updateMemoryDiagnostics();
}

// 7. Render graphical countdown overlay
function drawCountdownOverlay() {
    ctx.save();
    ctx.fillStyle = 'rgba(255, 42, 133, 0.85)';
    ctx.font = '80px Outfit';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
    ctx.shadowBlur = 15;
    ctx.fillText(state.countdownValue.toString(), elements.trackingCanvas.width / 2, elements.trackingCanvas.height / 2);
    ctx.restore();
}

// Process coordinates inside recording sequence
function handleDatasetRecording(coords) {
    state.recordingFrames.push(coords);
    
    const count = state.recordingFrames.length;
    elements.recordingFrameCount.innerText = `${count} / ${CONFIG.sequenceLength}`;
    elements.recordingProgressBar.style.width = `${(count / CONFIG.sequenceLength) * 100}%`;
    
    if (count === CONFIG.sequenceLength) {
        state.isRecording = false;
        
        // Save sequence to dataset array
        state.dataset.push({
            label: state.recordingLabel,
            sequence: state.recordingFrames
        });
        
        state.recordingFrames = [];
        
        // Hide progress UI and re-enable controls
        elements.recordingProgressWrapper.style.display = 'none';
        elements.btnRecordSequence.removeAttribute('disabled');
        
        // Flash canvas green as success signal
        flashCanvasSuccess();
        
        // Update Stats list
        updateDatasetStatsUI();
    }
}

// Canvas success indicator flash
function flashCanvasSuccess() {
    elements.trackingCanvas.style.boxShadow = '0 0 35px var(--status-active)';
    setTimeout(() => {
        elements.trackingCanvas.style.boxShadow = 'none';
    }, 350);
}

// Refresh sequence buffer UI telemetry
function updateBufferTelemetry() {
    const len = state.sequenceBuffer.length;
    elements.bufferText.innerText = `${len} / ${CONFIG.sequenceLength} frames`;
    elements.bufferFill.style.width = `${(len / CONFIG.sequenceLength) * 100}%`;
}

// Update stats listing in the Dataset Panel
function updateDatasetStatsUI() {
    // Count samples per label
    const counts = {};
    CONFIG.classes.forEach(cls => counts[cls] = 0);
    
    state.dataset.forEach(sample => {
        if (!counts[sample.label]) {
            counts[sample.label] = 0;
        }
        counts[sample.label]++;
    });
    
    // Update Total Counter
    elements.datasetTotalSamples.innerText = `${state.dataset.length} Samples`;
    
    // Update stats list
    elements.datasetStatsList.innerHTML = '';
    
    if (state.dataset.length === 0) {
        elements.datasetStatsList.innerHTML = '<li class="stats-item empty-state">No samples recorded yet.</li>';
        elements.btnDownloadDataset.setAttribute('disabled', 'true');
        elements.btnResetDataset.setAttribute('disabled', 'true');
        return;
    }
    
    // Enable actions
    elements.btnDownloadDataset.removeAttribute('disabled');
    elements.btnResetDataset.removeAttribute('disabled');
    
    Object.keys(counts).forEach(label => {
        const li = document.createElement('li');
        li.className = 'stats-item';
        li.innerHTML = `<i class="fa-solid fa-folder-open" style="color: var(--neon-cyan)"></i> <strong>${label}</strong>: ${counts[label]} samples`;
        elements.datasetStatsList.appendChild(li);
    });
}

// 8. Custom Cyberpunk Skeleton Drawer
function drawSkeleton(landmarks) {
    const width = elements.trackingCanvas.width;
    const height = elements.trackingCanvas.height;
    
    // A. Draw Connection Lines
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = 'rgba(137, 87, 255, 0.7)'; // Neon purple
    ctx.shadowBlur = 4;
    ctx.shadowColor = 'rgba(137, 87, 255, 0.5)';
    
    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
        const [startIdx, endIdx] = HAND_CONNECTIONS[i];
        const startPoint = landmarks[startIdx];
        const endPoint = landmarks[endIdx];
        
        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
    }
    
    // B. Draw Joints (Nodes)
    ctx.shadowBlur = 8;
    for (let i = 0; i < landmarks.length; i++) {
        const x = landmarks[i].x * width;
        const y = landmarks[i].y * height;
        
        const isFingertip = FINGERTIPS.includes(i);
        const radius = isFingertip ? 6.5 : 4.5;
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        
        if (isFingertip) {
            ctx.fillStyle = '#ff2a85'; // Neon pink tips
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.shadowColor = '#ff2a85';
            ctx.fill();
            ctx.stroke();
        } else if (i === 0) {
            ctx.fillStyle = '#00f2fe'; // Wrist anchor
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2.0;
            ctx.shadowColor = '#00f2fe';
            ctx.fill();
            ctx.stroke();
        } else {
            ctx.fillStyle = '#00f2fe';
            ctx.shadowColor = '#00f2fe';
            ctx.fill();
        }
    }
    
    ctx.shadowBlur = 0;
}

// 9. TensorFlow.js Inference Block
async function runInference() {
    const startTime = performance.now();
    
    // MEMORY MANAGEMENT: Manual tf.dispose() for async execution (tf.tidy does not support async/Promises)
    const inputTensor = tf.tensor3d([state.sequenceBuffer]);
    let predictionResults;
    try {
        const prediction = await executeModel(state.model, inputTensor);
        if (Array.isArray(prediction)) {
            predictionResults = prediction[0].squeeze().dataSync();
            prediction.forEach(t => t.dispose());
        } else {
            predictionResults = prediction.squeeze().dataSync();
            prediction.dispose();
        }
    } catch (err) {
        console.error("Inference execution failed:", err);
        inputTensor.dispose();
        return;
    }
    inputTensor.dispose();
    
    state.latency = Math.round(performance.now() - startTime);
    elements.hudLatency.innerText = `${state.latency} ms`;
    
    // Decode output class index
    let maxIdx = 0;
    let maxConf = 0;
    for (let i = 0; i < predictionResults.length; i++) {
        if (predictionResults[i] > maxConf) {
            maxConf = predictionResults[i];
            maxIdx = i;
        }
    }
    
    state.activePredictionWord = CONFIG.classes[maxIdx];
    state.currentConfidence = maxConf;
    
    updatePredictionUI();
    
    // Stability validation before appending words
    if (maxConf >= CONFIG.confidenceThreshold) {
        if (state.lastPredictedClass === maxIdx) {
            state.consecutivePredictions++;
            
            if (state.consecutivePredictions === CONFIG.stabilityFrames && state.cooldownCounter === 0) {
                appendWordToSentence(CONFIG.classes[maxIdx]);
                state.cooldownCounter = CONFIG.cooldownFrames;
            }
        } else {
            state.lastPredictedClass = maxIdx;
            state.consecutivePredictions = 1;
        }
    } else {
        state.consecutivePredictions = 0;
    }
}

// Update Active Word UI Elements
function updatePredictionUI() {
    if (state.activePredictionWord === 'WAITING...') {
        elements.activeWord.innerText = 'WAITING...';
        elements.confidencePercentage.innerText = '0%';
        elements.confidenceBar.style.width = '0%';
    } else {
        elements.activeWord.innerText = state.activePredictionWord.toUpperCase();
        const pct = Math.round(state.currentConfidence * 100);
        elements.confidencePercentage.innerText = `${pct}%`;
        elements.confidenceBar.style.width = `${pct}%`;
        
        if (state.currentConfidence > 0.8) {
            elements.confidenceBar.style.background = 'linear-gradient(90deg, var(--neon-blue) 0%, var(--neon-cyan) 100%)';
        } else {
            elements.confidenceBar.style.background = 'linear-gradient(90deg, var(--neon-purple) 0%, var(--neon-pink) 100%)';
        }
    }
}

// Append stable word to text area
function appendWordToSentence(word) {
    let currentText = elements.sentenceHistory.value.trim();
    
    if (currentText.length > 0) {
        currentText += " " + word;
    } else {
        currentText = word;
    }
    
    elements.sentenceHistory.value = currentText;
    elements.sentenceHistory.scrollTop = elements.sentenceHistory.scrollHeight;
    
    elements.sentenceHistory.style.boxShadow = '0 0 15px rgba(0, 255, 135, 0.2)';
    setTimeout(() => {
        elements.sentenceHistory.style.boxShadow = 'none';
    }, 500);
}

// Diagnostics tracker for memory usage
function updateMemoryDiagnostics() {
    if (window.tf) {
        const info = tf.memory();
        elements.memoryStats.innerText = `${info.numTensors} Tensors (${Math.round(info.numBytes / 1024)} KB)`;
    }
}

// 10. Webcam Controllers & Streams
async function toggleWebcam() {
    if (state.isCameraActive) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    try {
        console.log("Attempting to start webcam...");
        elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing...';
        elements.btnToggleCamera.setAttribute('disabled', 'true');
        
        const constraints = {
            video: {
                facingMode: 'user',
                width: 640,
                height: 480
            },
            audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.webcamFeed.srcObject = stream;
        
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
        elements.statusWebcam.setAttribute('data-status', 'active');
        elements.statusWebcam.querySelector('.status-text').innerText = 'ACTIVE';
        
        elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-power-off"></i> Stop Webcam';
        elements.btnToggleCamera.removeAttribute('disabled');
        elements.btnToggleCamera.style.background = 'linear-gradient(135deg, var(--neon-pink) 0%, #e01b5d 100%)';
        elements.btnToggleCamera.style.boxShadow = '0 4px 15px rgba(255, 42, 133, 0.3)';
    } catch (err) {
        console.error("Camera activation failure:", err);
        alert("Unable to access webcam. Please check browser permissions and try again.");
        
        elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Webcam';
        elements.btnToggleCamera.removeAttribute('disabled');
        stopWebcam();
    }
}

function stopWebcam() {
    console.log("Stopping webcam stream...");
    
    if (cameraHelper) {
        cameraHelper.stop();
        cameraHelper = null;
    }
    
    if (elements.webcamFeed.srcObject) {
        const stream = elements.webcamFeed.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        elements.webcamFeed.srcObject = null;
    }
    
    state.isCameraActive = false;
    state.sequenceBuffer = [];
    updateBufferTelemetry();
    
    elements.statusWebcam.setAttribute('data-status', 'inactive');
    elements.statusWebcam.querySelector('.status-text').innerText = 'OFF';
    
    elements.statusDetection.setAttribute('data-status', 'none');
    elements.statusDetection.querySelector('.status-text').innerText = 'NONE';
    
    elements.btnToggleCamera.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Webcam';
    elements.btnToggleCamera.style.background = 'linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-cyan) 100%)';
    elements.btnToggleCamera.style.boxShadow = '0 4px 15px rgba(79, 172, 254, 0.3)';
    
    setupCanvas();
}

// 11. Event Listeners Config
function setupEventListeners() {
    // Camera toggle
    elements.btnToggleCamera.addEventListener('click', toggleWebcam);
    
    // Translation Pause toggle
    elements.btnToggleTranslation.addEventListener('click', () => {
        state.isTranslationPaused = !state.isTranslationPaused;
        if (state.isTranslationPaused) {
            elements.btnToggleTranslation.innerHTML = '<i class="fa-solid fa-play"></i> Resume Translation';
            elements.btnToggleTranslation.style.background = 'rgba(255, 211, 29, 0.1)';
            elements.btnToggleTranslation.style.color = 'var(--status-warning)';
            elements.btnToggleTranslation.style.borderColor = 'rgba(255, 211, 29, 0.3)';
        } else {
            elements.btnToggleTranslation.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Translation';
            elements.btnToggleTranslation.removeAttribute('style');
        }
    });
    
    // Clear History
    elements.btnClearHistory.addEventListener('click', () => {
        elements.sentenceHistory.value = '';
    });
    
    // TTS Voice Engine
    elements.btnTts.addEventListener('click', () => {
        const text = elements.sentenceHistory.value.trim();
        if (text.length === 0) return;
        
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.05;
            window.speechSynthesis.speak(utterance);
        } else {
            alert("Your browser does not support text-to-speech features.");
        }
    });

    // Dataset Builder Event: Add Custom Class Label
    elements.btnAddClass.addEventListener('click', () => {
        const customName = elements.inputCustomClass.value.trim();
        if (customName) {
            // Check if already in options
            let exists = false;
            for (let i = 0; i < elements.selectRecordClass.options.length; i++) {
                if (elements.selectRecordClass.options[i].value === customName) {
                    exists = true;
                    elements.selectRecordClass.selectedIndex = i;
                    break;
                }
            }
            
            if (!exists) {
                const opt = document.createElement('option');
                opt.value = customName;
                opt.innerText = customName;
                elements.selectRecordClass.appendChild(opt);
                elements.selectRecordClass.value = customName;
                
                // Add to classes array if not in it
                if (!CONFIG.classes.includes(customName)) {
                    CONFIG.classes.push(customName);
                }
            }
            
            // Refresh selector list and classes counter UI
            syncClassSelector();
            elements.inputCustomClass.value = '';
        }
    });

    // Dataset Builder Event: Record Sample Sequence
    elements.btnRecordSequence.addEventListener('click', () => {
        if (!state.isCameraActive) {
            alert("Please start the webcam first before attempting to record gestures.");
            return;
        }
        
        state.recordingLabel = elements.selectRecordClass.value;
        elements.btnRecordSequence.setAttribute('disabled', 'true');
        
        // Reset recording variables
        state.recordingFrames = [];
        elements.recordingProgressWrapper.style.display = 'block';
        elements.recordingStateLabel.innerText = `Get ready... Starting in 3s`;
        elements.recordingFrameCount.innerText = '0 / 30';
        elements.recordingProgressBar.style.width = '0%';
        
        // Start 3-second countdown
        state.countdownValue = 3;
        state.countdownInterval = setInterval(() => {
            state.countdownValue--;
            if (state.countdownValue > 0) {
                elements.recordingStateLabel.innerText = `Get ready... Starting in ${state.countdownValue}s`;
            } else {
                clearInterval(state.countdownInterval);
                state.countdownInterval = null;
                // Initiate active frame collection
                state.isRecording = true;
                elements.recordingStateLabel.innerText = `Recording "${state.recordingLabel}"...`;
            }
        }, 1000);
    });

    // Dataset Builder Event: Download JSON Dataset
    elements.btnDownloadDataset.addEventListener('click', () => {
        if (state.dataset.length === 0) return;
        
        const jsonStr = JSON.stringify(state.dataset, null, 4);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Download trigger
        const a = document.createElement('a');
        a.href = url;
        a.download = 'dataset.json';
        document.body.appendChild(a);
        a.click();
        
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`Downloaded dataset containing ${state.dataset.length} samples.`);
    });

    // Dataset Builder Event: Reset local collection
    elements.btnResetDataset.addEventListener('click', () => {
        if (confirm("Are you sure you want to delete all currently recorded dataset samples?")) {
            state.dataset = [];
            updateDatasetStatsUI();
        }
    });
}
