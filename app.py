import os
import time
import base64
import json
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

app = Flask(__name__)

# ================= ENV =================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS = {}
ESP_LOCK = Lock()

# ================= GITHUB CONFIG =================
GITHUB_REPO = "onlykartika/ESP32-CAM"  # Ubah ke repo yang benar kalau Kolam B
GITHUB_FOLDER = "images"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

# ================= LOAD / SAVE RESULTS =================
def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                ESP_RESULTS = json.load(f)
                print("[INFO] Loaded esp_results from local file")
                return
        except Exception as e:
            print(f"[ERROR] Failed load local: {e}")

    # Fallback GitHub
    try:
        res = requests.get(f"{GITHUB_API_ROOT}/esp_results.json", headers=GITHUB_HEADERS, timeout=10)
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode()
            ESP_RESULTS = json.loads(content)
            with open(ESP_RESULTS_FILE, "w") as f:
                json.dump(ESP_RESULTS, f, indent=2)
            print("[INFO] Loaded and saved esp_results from GitHub")
            return
    except Exception as e:
        print(f"[WARN] Failed load from GitHub: {e}")

    ESP_RESULTS = {}

def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)
        print("[INFO] Saved esp_results locally")
    except Exception as e:
        print(f"[ERROR] Save local failed: {e}")

load_esp_results()

# ================= ROBOFLOW =================
rf_client = None
def get_rf_client():
    global rf_client
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",  # Ubah ke detect kalau pakai model detect biasa
            api_key=ROBOFLOW_API_KEY
        )
    return rf_client

WORKSPACE_NAME = "my-workspace-rrwxa"
WORKFLOW_ID = "detect-count-and-visualize"
TARGET_LABEL = "male"  # Ganti kalau target female/indukan
CONF_THRESHOLD = 0.4

# ================= HEALTH =================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "AI server running - Female/Male Detection Ready"})

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image data"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")
    # Ambil timestamp dari ESP kalau dikirim via header, fallback server time
    esp_timestamp = request.headers.get("X-Timestamp")
    if esp_timestamp:
        try:
            timestamp = int(esp_timestamp)
        except:
            timestamp = int(time.time())
    else:
        timestamp = int(time.time())

    filename = f"{esp_id}_{timestamp}.jpg"
    print(f"[INFO] Upload dari {esp_id} | timestamp: {timestamp} | size: {len(request.data)} bytes")

    # Simpan sementara
    temp_path = filename
    try:
        with open(temp_path, "wb") as f:
            f.write(request.data)
    except Exception as e:
        return jsonify({"error": "save temp image failed", "detail": str(e)}), 500

    # ===== ROBOFLOW WORKFLOW =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=False
        )
        print("[DEBUG] Raw Roboflow result:", json.dumps(result, indent=2, default=str))
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": "roboflow workflow failed", "detail": str(e)}), 500

    # Parse predictions (lebih robust)
    predictions = []
    try:
        if isinstance(result, list) and result:
            for item in result:
                if "predictions" in item:
                    predictions.extend(item["predictions"])
        elif isinstance(result, dict):
            if "predictions" in result:
                predictions = result["predictions"]
            elif "result" in result and isinstance(result["result"], list):
                predictions = result["result"]
    except Exception as e:
        print(f"[WARN] Parse predictions failed: {e}")

    filtered = [
        p for p in predictions
        if p.get("class", "").lower() == TARGET_LABEL.lower() and p.get("confidence", 0) >= CONF_THRESHOLD
    ]
    detected_count = len(filtered)

    # ===== UPLOAD IMAGE TO GITHUB =====
    github_success = False
    try:
        with open(temp_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        folder_path = f"{GITHUB_FOLDER}/esp_{esp_id.split('_')[-1] if '_' in esp_id else esp_id}"
        put_url = f"{GITHUB_API_ROOT}/{folder_path}/{filename}"

        # Cek SHA kalau file sudah ada (opsional, tapi aman)
        get_res = requests.get(put_url, headers=GITHUB_HEADERS, timeout=5)
        sha = get_res.json().get("sha") if get_res.status_code == 200 else None

        payload = {
            "message": f"upload from esp {esp_id} ({filename})",
            "content": content_b64
        }
        if sha:
            payload["sha"] = sha

        res = requests.put(put_url, headers=GITHUB_HEADERS, json=payload, timeout=15)
        if res.status_code in (200, 201):
            github_success = True
            print("[INFO] Image uploaded to GitHub successfully")
        else:
            print(f"[WARN] GitHub upload failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[ERROR] GitHub upload exception: {e}")

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # ===== UPDATE RESULTS (MERGE, BUKAN OVERWRITE) =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": detected_count,
            "last_update": timestamp * 1000  # pakai timestamp gambar, bukan waktu server
        }
        save_esp_results()

        # Backup ke GitHub esp_results.json
        try:
            json_content = json.dumps(ESP_RESULTS, indent=2).encode()
            content_b64 = base64.b64encode(json_content).decode()

            json_path = f"{GITHUB_API_ROOT}/esp_results.json"
            get_res = requests.get(json_path, headers=GITHUB_HEADERS, timeout=10)
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            payload = {
                "message": f"Update results from {esp_id} - {time.strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64
            }
            if sha:
                payload["sha"] = sha

            put_res = requests.put(json_path, headers=GITHUB_HEADERS, json=payload, timeout=15)
            if put_res.status_code in (200, 201):
                print("[INFO] esp_results.json updated on GitHub")
            else:
                print(f"[WARN] GitHub JSON update failed: {put_res.text}")
        except Exception as e:
            print(f"[WARN] GitHub JSON backup failed: {e}")

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "detected_this_esp": detected_count,
        "total_all_esp": total_all,
        "per_esp": ESP_RESULTS,
        "objects": filtered,
        "github_image_success": github_success
    })

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(v["count"] for v in ESP_RESULTS.values()),
            "per_esp": ESP_RESULTS
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
