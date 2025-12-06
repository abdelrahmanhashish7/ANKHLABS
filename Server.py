from flask import Flask, request, jsonify, send_file
import io
import base64
from zeroconf import ServiceInfo, Zeroconf
import socket

app = Flask(__name__)

# Buffers and latest values
ecg_buffer = []          # [{"ecg": value, "timestamp": X}, ...]
latest_plot = None       # Base64 string of ECG plot
latest_rr = None         # Latest respiration rate
latest_ecg_numbers = []  # Latest ECG numbers sent from NK
resp_rate_history =[]
glucose_buffer = [] 
resp_rate_buffer = []
resp_last_time = None# list of {"glucose": value, "timestamp": X}
latest_glucose = {
    "value": None,
    "timestamp": None  # when the value was received
}     # most recent glucose reading
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

local_ip = get_local_ip()
desc = {'path': '/'}
info = ServiceInfo(
    "_http._tcp.local.",
    "flaskserver._http._tcp.local.",
    addresses=[socket.inet_aton(local_ip)],
    port=5000,
    properties=desc,
    server="flaskserver.local."
)

zeroconf = Zeroconf()
zeroconf.register_service(info)

print(f"Flask server advertised as http://flaskserver.local:5000")
# -----------------------------
# Raw ECG data buffer
# -----------------------------
@app.route('/data', methods=['POST'])
def receive_data():
    global latest_ecg_numbers
    data = request.get_json()
    if not data or "ecg" not in data:
        return jsonify({"status": "error", "message": "bad JSON"}), 400


    ecg = data.get("ecg")
    glucose = data.get("glucose")
    timestamp = data.get("timestamp")

    # Always store ECG
    ecg_buffer.append({"ecg": ecg, "timestamp": timestamp})
    latest_ecg_numbers.append(ecg)

    # Handle glucose ONLY when it's real (not null)
    if glucose is not None:
       latest_glucose["value"] = glucose
       latest_glucose["timestamp"] = timestamp
       glucose_buffer.append({"glucose": glucose, "timestamp": timestamp})
       print(f"New 1-minute glucose: {glucose}")

    print("Received:", data)
    return jsonify({"status": "ok"}), 200

@app.route('/ecg', methods=['GET'])
def get_ecg():
    # Return only last ~1 second (250 samples)
    return jsonify(ecg_buffer[-500:])

@app.route('/clear', methods=['POST'])
def clear_buffer():
    ecg_buffer.clear()
    latest_ecg_numbers.clear()
    return jsonify({"status": "cleared"})

# -----------------------------
# ECG Plot
# -----------------------------
@app.route('/ecg_plot', methods=['GET'])
def show_ecg_plot():
    global latest_plot
    if latest_plot is None:
        return "No plot available yet", 404
    img_bytes = base64.b64decode(latest_plot)
    return send_file(io.BytesIO(img_bytes), mimetype='image/png')

@app.route('/ecg_plot', methods=['POST'])
def receive_ecg_plot():
    global latest_plot
    data = request.get_json()
    latest_plot = data.get("plot")
    return jsonify({"status": "ok"}), 200

# -----------------------------
# Respiration Rate
# -----------------------------
@app.route('/resp_rate', methods=['GET'])
def show_resp_rate():
    global latest_rr
    if latest_rr is None:
        return jsonify({"resp_rate": None}), 404
    return jsonify({"resp_rate": latest_rr})

@app.route('/resp_rate', methods=['POST'])
def receive_resp_rate():
    global latest_rr, resp_rate_buffer, resp_window_start, resp_rate_history

    data = request.get_json()
    sample = data.get("resp_rate")

    if sample is None:
        return jsonify({"status": "error", "message": "no resp_rate provided"}), 400

    now = time.time()

    # Start 1-minute collection window if first sample
    if resp_window_start is None:
        resp_window_start = now
        resp_rate_buffer = []  # reset buffer

    resp_rate_buffer.append(sample)

    # If 60 seconds passed → compute 1-minute average
    if now - resp_window_start >= 60:

        avg_rr = sum(resp_rate_buffer) / len(resp_rate_buffer)
        latest_rr = avg_rr

        # SAVE FOR HISTORY (this is what you asked to preserve)
        resp_rate_history.append(avg_rr)

        print(f"[RESP] 1-minute AVG: {avg_rr}")

        # reset window
        resp_rate_buffer = []
        resp_window_start = None

        return jsonify({
            "status": "done",
            "average_resp_rate": avg_rr
        })

    # Still collecting
    return jsonify({
        "status": "collecting",
        "samples_collected": len(resp_rate_buffer)
    })


# -----------------------------
# ECG Numbers (raw values from NK or /data)
# -----------------------------
@app.route('/ecgnumbers', methods=['GET'])
def show_ecg_numbers():
    global latest_ecg_numbers
    if not latest_ecg_numbers:
        return jsonify({"numbers": []}), 404
    return jsonify({"numbers": latest_ecg_numbers})
    #return jsonify({"numbers": latest_ecg_numbers[-250:]})

@app.route('/ecgnumbers', methods=['POST'])
def receive_ecg_numbers():
    global latest_ecg_numbers
    data = request.get_json()
    latest_ecg_numbers = data.get("numbers", [])
    return jsonify({"status": "ok"}), 200
@app.route('/resp_history', methods=['GET'])
def show_resp_history():
    global resp_rate_history
    if not resp_rate_history:
        return jsonify({"resp_history": []}), 404
    return jsonify({"resp_history": resp_rate_history})

@app.route('/resp_history', methods=['POST'])
def receive_resp_history():
    global resp_rate_history
    data = request.get_json()
    resp_rate_history = data.get("resp_history", [])
    return jsonify({"status": "ok"}), 200
# -----------------------------
# Test route (optional)
# -----------------------------
@app.route("/test")
def home():
    return "hello world"

@app.route('/glucose', methods=['GET'])
def show_glucose_numbers():
    global latest_glucose
    if latest_glucose["value"] is None:
        return jsonify({"glucose": None}), 404

    # Return the value and timestamp
    return jsonify({
        "glucose": latest_glucose["value"],
        "timestamp": latest_glucose["timestamp"]
    })

@app.route('/glucose', methods=['POST'])
def receive_glucose_numbers():
    global latest_glucose, glucose_buffer
    data = request.get_json()
    glucose = data.get("glucose")
    timestamp = data.get("timestamp")  # optional, can come from ESP32

    # Only update if glucose is real (not null)
    if glucose is not None:
        latest_glucose["value"] = glucose
        latest_glucose["timestamp"] = timestamp
        glucose_buffer.append({"glucose": glucose, "timestamp": timestamp})
        print(f"Updated glucose via POST: {glucose}")
    return jsonify({"status": "ok"}), 200

@app.route('/glucose_history', methods=['GET'])
def glucose_history():
    if len(glucose_buffer) == 0:
        return jsonify({"glucose_history": []}), 404

    # Return all 1-minute glucose readings
    return jsonify({
        "glucose_history": glucose_buffer
    })




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
