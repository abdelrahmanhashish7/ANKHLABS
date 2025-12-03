from flask import Flask, request, jsonify, send_file
import io
import base64
import threading
import neurokit  # <-- our processing module

app = Flask(__name__)

# -----------------------------
# Buffers & shared state
# -----------------------------
ecg_buffer = []            # [{"ecg": value, "timestamp": X}, ...]
glucose_buffer = []        # glucose history
latest_glucose = {"value": None, "timestamp": None}

latest_plot = None         # deprecated (now inside neurokit)
latest_rr = None           # deprecated (inside neurokit)
latest_ecg_numbers = []    # deprecated

# -----------------------------
# Receive sensor data
# -----------------------------
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    if not data or "ecg" not in data:
        return jsonify({"status": "error", "message": "bad JSON"}), 400

    ecg = data.get("ecg")
    glucose = data.get("glucose")
    timestamp = data.get("timestamp")

    # store ECG
    ecg_buffer.append({"ecg": ecg, "timestamp": timestamp})

    # optional glucose
    if glucose is not None:
        latest_glucose["value"] = glucose
        latest_glucose["timestamp"] = timestamp
        glucose_buffer.append({"glucose": glucose, "timestamp": timestamp})

    return jsonify({"status": "ok"}), 200

# -----------------------------
# Get ECG buffer (raw)
# -----------------------------
@app.route('/ecg', methods=['GET'])
def get_ecg():
    return jsonify(ecg_buffer[-500:])  # last 500 samples

# -----------------------------
# Clear buffers
# -----------------------------
@app.route('/clear', methods=['POST'])
def clear_buffer():
    ecg_buffer.clear()
    glucose_buffer.clear()
    return jsonify({"status": "cleared"})

# -----------------------------
# Respiration Rate (processed)
# -----------------------------
@app.route('/resp_rate', methods=['GET'])
def show_resp_rate():
    return jsonify({"resp_rate": neurokit.latest_rr})

# -----------------------------
# ECG Plot (processed)
# -----------------------------
@app.route('/ecg_plot', methods=['GET'])
def show_ecg_plot():
    if neurokit.latest_plot is None:
        return "No plot", 404
    img_bytes = base64.b64decode(neurokit.latest_plot)
    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

# -----------------------------
# Processed ECG numbers
# -----------------------------
@app.route('/ecgnumbers', methods=['GET'])
def show_ecg_numbers():
    return jsonify({"numbers": neurokit.latest_ecg_numbers})

# -----------------------------
# Glucose
# -----------------------------
@app.route('/glucose', methods=['GET'])
def show_glucose():
    if latest_glucose["value"] is None:
        return jsonify({"glucose": None}), 404

    return jsonify({
        "glucose": latest_glucose["value"],
        "timestamp": latest_glucose["timestamp"]
    })

@app.route('/glucose_history', methods=['GET'])
def show_glucose_history():
    return jsonify({"glucose_history": glucose_buffer})

# -----------------------------
# Test route
# -----------------------------
@app.route('/test')
def home():
    return "Server running"

# -----------------------------
# Start NeuroKit processing thread
# -----------------------------
def start_processing_thread():
    thread = threading.Thread(
        target=neurokit.neurokit_worker,
        args=(ecg_buffer,),
        daemon=True
    )
    thread.start()
    print("[Server] NeuroKit worker started.")

start_processing_thread()

# -----------------------------
# Run Flask server
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
