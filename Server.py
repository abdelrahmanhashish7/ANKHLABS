from flask import Flask, request, jsonify, send_file
import io
import base64
from zeroconf import ServiceInfo, Zeroconf
import socket
import subprocess
import signal
import os
import threading

app = Flask(__name__)
current_wifi = {"ssid": None, "password": None}
process = None
# Buffers and latest values
ecg_buffer = []          # [{"ecg": value, "timestamp": X}, ...]
latest_plot = None       # Base64 string of ECG plot
latest_rr = None         # Latest respiration rate
latest_ecg_numbers = []  # Latest ECG numbers sent from NK
resp_rate_history =[]
glucose_buffer = [] # list of {"glucose": value, "timestamp": X}
latest_glucose = {
    "value": None,
    "timestamp": None  # when the value was received
}     # most recent glucose reading
FIRST_TIME_FILE = r"C:\Users\nadim\Desktop\FinalGlucoECG\src\FinalGlucoECG.cpp\first_time.txt"
SKETCH_FOLDER = r"C:\Users\nadim\Desktop\FinalGlucoECG\src\FinalGlucoECG.cpp"
INO_FILE = os.path.join(SKETCH_FOLDER, "FinalGlucoECG.cpp")
#TEST_MODE: True
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


@app.route('/wifi_config', methods=['POST'])
def wifi_config():
    global current_wifi
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "no JSON received"}), 400

    ssid = data.get("ssid")
    password = data.get("password")

    if not ssid or not password:
        return jsonify({"status": "error", "message": "ssid or password missing"}), 400

    current_wifi["ssid"] = ssid
    current_wifi["password"] = password

    print(f"Stored Wi-Fi credentials → SSID={ssid}, PASSWORD={password}")

    return jsonify({"status": "ok", "message": "wifi stored"}), 200
@app.route('/upload_esp', methods=['POST'])
def upload_esp():
    try:
        global current_wifi

        data = request.get_json()
        com_port = data.get("port", "COM4")

        first_time = not os.path.exists(FIRST_TIME_FILE)

        # Use stored Wi-Fi credentials on first time
        if first_time:
            ssid = current_wifi.get("ssid")
            password = current_wifi.get("password")

            if not ssid or not password:
                return jsonify({
                    "status": "error",
                    "message": "Wi-Fi not set. Call /wifi_config before /upload_esp."
                }), 400

            print(f"Injecting Wi-Fi into PlatformIO sketch: SSID={ssid}")

            # Insert into .cpp
            with open(INO_FILE, "r") as f:
                code = f.read()

            code = code.replace("WIFI_SSID_PLACEHOLDER", ssid)
            code = code.replace("WIFI_PASSWORD_PLACEHOLDER", password)

            with open(INO_FILE, "w") as f:
                f.write(code)

            print("Wi-Fi placeholders replaced for first-time setup.")

        # Run Arduino CLI in background
        def pio_upload():
         cmd = ["platformio", "run", "--target", "upload", f"--upload-port={com_port}"]
         proc = subprocess.Popen(cmd, cwd=SKETCH_FOLDER, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
         out, err = proc.communicate()
         if proc.returncode == 0:
            print("Upload successful:\n", out)
         else:
            print("Upload failed:\n", err)

        threading.Thread(target=pio_upload, daemon=True).start()

        return jsonify({"status": "upload_started", "message": "ESP upload started in background"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# Debug route: show current Wi-Fi
# -----------------------------
@app.route('/debug_wifi', methods=['GET'])
def debug_wifi():
    first_time_exists = os.path.exists(FIRST_TIME_FILE)
    ssid = None
    password = None

    # If first-time, read placeholders from the .ino file
    if first_time_exists:
        # After first-time upload, credentials are now in ESP runtime via AP mode
        return jsonify({"status": "first-time upload done. Use ESP AP to check Wi-Fi."})
    else:
        # Extract current placeholders from the .ino file (optional)
        with open(INO_FILE, "r") as f:
            code = f.read()
        # Try to read the replaced SSID and password (naive approach)
        import re
        ssid_match = re.search(r'WIFI_SSID_PLACEHOLDER', code)
        pass_match = re.search(r'WIFI_PASSWORD_PLACEHOLDER', code)
        ssid = ssid_match.group(0) if ssid_match else "unknown"
        password = pass_match.group(0) if pass_match else "unknown"

    return jsonify({
        "first_time_done": first_time_exists,
        "ssid": ssid,
        "password": password
    })

@app.route('/esp_wifi', methods=['POST'])
def receive_esp_wifi():
    global current_esp_wifi
    data = request.get_json()
    if not data or "ssid" not in data or "password" not in data:
        return jsonify({"status": "error", "message": "ssid or password missing"}), 400
    
    current_esp_wifi["ssid"] = data["ssid"]
    current_esp_wifi["password"] = data["password"]

    print(f"ESP reported Wi-Fi: {current_esp_wifi}")
    return jsonify({"status": "ok"}), 200

# START PROCESSING SCRIPT
@app.route('/start_processing', methods=['POST'])
def start_processing():
    global latest_ecg_numbers, latest_plot, latest_rr

    if not latest_ecg_numbers:
        return jsonify({"status": "error", "message": "no ECG data available"}), 400

    # You can decide how many samples to pass. Here we pass all collected numbers.
    result = NeuroKIt.process_data(latest_ecg_numbers, fs=50, window_sec=30)

    if result.get("status") != "ok":
        return jsonify({"status": "error", "message": result.get("message")}), 500

    latest_rr = result.get("resp_rate")
    latest_plot = result.get("plot")   # base64 PNG
    # optionally save edr / rr series into history
    # resp_rate_history.append(latest_rr)

    return jsonify({
        "status": "processing done",
        "resp_rate": latest_rr,
        "plot_sent": bool(latest_plot)
    })
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
    global latest_rr
    data = request.get_json()
    latest_rr = data.get("resp_rate")
    return jsonify({"status": "ok"}), 200

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
