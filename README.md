# Antigravity: Real-Time Client-Side Sign Language Translator

**Antigravity** is a highly optimized, 100% client-side sign language translator running entirely inside the web browser. It leverages **MediaPipe Hands JS** for real-time landmark feature extraction, and **TensorFlow.js** for high-performance LSTM model inference.

With zero backend dependencies, it ensures user data privacy, low latency, and portability.

---

## File Structure

```
gestures/
├── index.html          # Frontend page with MediaPipe and TensorFlow.js CDN scripts
├── style.css           # Premium glassmorphic HUD dark styling
├── app.js              # Camera streaming, coordinate queues, and inference logic
├── train.py            # Python LSTM training script using TensorFlow/Keras
├── convert_model.sh    # Script to convert the Keras model (.h5) into TF.js format
├── requirements.txt    # Python dependencies list
└── README.md           # Setup and execution guide (This file)
```

---

## Installation & Setup

### Phase 1: Python Training Environment
To train the LSTM neural network on landmark data:

1. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   Run the training script to generate `model.h5`:
   ```bash
   python3 train.py
   ```
   *Note: This script generates synthetic data of shape `(samples, 30 frames, 63 coordinates)` matching the input shape of 21 landmarks x 3 axes (X, Y, Z) over a rolling 30-frame window.*

---

### Phase 2: Model Conversion (Keras to TF.js)
To translate the `.h5` model file into browser-interpretable format (`model.json` + weight binaries):

1. **Convert the model**:
   Run the conversion script:
   ```bash
   chmod +x convert_model.sh
   ./convert_model.sh
   ```
   Alternatively, run the command directly in the terminal:
   ```bash
   tensorflowjs_converter --input_format=keras model.h5 tfjs_model
   ```

3. **Output Files**:
   This produces a folder named `tfjs_model/` inside your workspace containing:
   - `model.json` (Model topology and weights metadata)
   - `group1-shard1of1.bin` (Binary weights data)

---

### Phase 3: Launching the Frontend Application
Since the browser restricts reading local files directly from the filesystem due to CORS (Cross-Origin Resource Sharing), **you must host the files using a local server** (rather than opening `index.html` via double-clicking).

1. **Run a local python server**:
   Start Python's built-in lightweight HTTP server in the project directory:
   ```bash
   python3 -m http.server 8000
   ```

2. **Open the web app**:
   Navigate to the following address in your browser (preferably Chrome or Edge for optimal WebGL performance):
   [http://localhost:8000](http://localhost:8000)

3. **Operate**:
   - Allow camera access.
   - Click **Start Webcam**.
   - Show your hand to the camera to see the glowing skeleton tracker and real-time predictions.

---

## How it Works (Under the Hood)

1. **Feature Extraction (MediaPipe)**: MediaPipe identifies 21 joints of the hand in 3D space. Every frame, we extract the $(X, Y, Z)$ normalized values for each of the 21 joints, flattening them into a single list of $63$ floats.
2. **Rolling Queue Buffer (Vanilla JS)**: We feed these coordinate lists into a rolling array buffer. Once the array holds the last $30$ consecutive frames of data, it structures a tensor of shape `[1, 30, 63]`.
3. **Memory Management (TF.js)**: Continuous frame processing poses memory leak risks. To prevent this, predictions are executed inside a `tf.tidy()` wrapper, which auto-disposes intermediate tensors. We display active tensors on-screen under telemetry metrics to verify memory safety.
4. **Hysteresis Filtration**: To avoid flickering text, predictions are filtered using a stabilization threshold. A gesture must be consistently predicted for $7$ consecutive frames before the word is appended to the assembled sentence box.
5. **Speech Synthesis**: The Web Speech API is wired in to speak the accumulated text out loud.

---

## Adding Custom Gestures

To train the translator on your own custom sign language gestures:
1. **Gather Sequences**: Save actual MediaPipe coordinate outputs into CSV or numpy structures (`shape=(30, 63)`) for each of your gestures.
2. **Modify `train.py`**:
   - Update `num_classes` with your class count.
   - Replace the synthetic data generation with your custom dataset loading code.
   - Save your model to `model.h5`.
3. **Modify `app.js`**:
   - In `CONFIG.classes`, replace the dummy labels with your custom gesture names:
     ```javascript
     classes: ['A', 'B', 'C', 'Hello', 'Thank You']
     ```
4. **Re-convert & Reload**: Re-run the converter and refresh your browser!
