import os
import time
import base64
import json
import requests
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from threading import Lock

# ================= FLASK APP =================
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
GITHUB_REPO = "onlykartika/indukan-detect"
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
                print("Loaded results from local file")
                return ESP_RESULTS
        except Exception as e:
            print(f"Failed to load local file: {e}")

    # Coba ambil dari GitHub kalau local gagal/kosong
    try:
        res = requests.get(f"{GITHUB_API_ROOT}/esp_results.json", headers=GITHUB_HEADERS, timeout=10)
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode('utf-8')
            ESP_RESULTS = json.loads(content)
            save_esp_results()  # sync ke local
            print("Loaded results from GitHub")
            return ESP_RESULTS
    except Exception as e:
        print(f"Failed to load from GitHub: {e}")

    ESP_RESULTS = {}
    return ESP_RESULTS

def save_esp_results():
    with open(ESP_RESULTS_FILE, "w") as f:
        json.dump(ESP_RESULTS, f, indent=2)
    print("Saved results to local file")

ESP_RESULTS = load_esp_results()

# ================= GITHUB FOLDER HELPER =================
def ensure_github_folder(esp_id):
    """
    Pastikan folder images/<esp_id>/ ada di GitHub.
    Kalau belum ada, buat dengan file .gitkeep (kosong).
    """
    folder_path = f"{GITHUB_FOLDER}/{esp_id}"
    url = f"{GITHUB_API_ROOT}/{folder_path}"

    # Cek apakah folder sudah ada
    res = requests.get(url, headers=GITHUB_HEADERS, timeout=10)
    if res.status_code == 200:
        return True  # folder sudah ada

    if res.status_code != 404:
        print(f"Unexpected status when checking folder {folder_path}: {res.status_code} - {res.text}")
        return False

    # Buat folder dengan .gitkeep
    gitkeep_url = f"{url}/.gitkeep"
    payload = {
        "message": f"Create folder for ESP {esp_id}",
        "content": ""  # file kosong (base64 dari empty string)
    }

    put_res = requests.put(gitkeep_url, headers=GITHUB_HEADERS, json=payload, timeout=15)
    if put_res.status_code in (200, 201):
        print(f"Successfully created folder: {folder_path}")
        return True
    else:
        print(f"Failed to create folder {folder_path}: {put_res.status_code} - {put_res.text}")
        return False

# ================= ROBOFLOW =================
rf_client = None

def get_rf_client():
    global rf_client
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
    return rf_client

WORKSPACE_NAME = "my-workspace-rrwxa"
WORKFLOW_ID = "detect-count-and-visualize"
TARGET_LABEL = "female"
CONF_THRESHOLD = 0.6

# ================= HEALTH =================
@app.route("/", methods=["GET"])
def health():
    return "AI server running (Female Detection Mode) - Ready for ESP uploads"

# ================= IMAGE UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image data"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")
    filename = f"{esp_id}_{int(time.time())}.jpg"

    print(f"Received image from {esp_id}, size: {len(request.data)} bytes")

    # Simpan sementara untuk inference
    with open(filename, "wb") as f:
        f.write(request.data)

    # ===== ROBOFLOW INFERENCE =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
        print("Roboflow inference completed")
    except Exception as e:
        os.remove(filename)
        print(f"Roboflow error: {e}")
        return jsonify({"error": "roboflow inference failed", "detail": str(e)}), 500

    # ===== UPLOAD TO GITHUB =====
    try:
        # Pastikan folder esp_id ada
        if ensure_github_folder(esp_id):
            with open(filename, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode('utf-8')

            file_path = f"{GITHUB_FOLDER}/{esp_id}/{filename}"
            put_url = f"{GITHUB_API_ROOT}/{file_path}"

            res = requests.put(
                put_url,
                headers=GITHUB_HEADERS,
                json={
                    "message": f"Upload from {esp_id}: {filename}",
                    "content": content_b64
                },
                timeout=15
            )

            if res.status_code in (200, 201):
                print(f"Uploaded to GitHub: {file_path}")
            else:
                print(f"GitHub upload failed ({res.status_code}): {res.text}")
        else:
            print(f"Skipping GitHub upload for {esp_id} - folder creation failed")
    except Exception as e:
        print(f"Exception during GitHub upload: {e}")

    # Bersihkan file lokal
    if os.path.exists(filename):
        os.remove(filename)

    # ===== PARSE RESULT =====
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    filtered = []
    for p in predictions:
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0
        if label and label.lower() == TARGET_LABEL.lower() and conf >= CONF_THRESHOLD:
            filtered.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    detected_count = len(filtered)

    # ===== UPDATE STATE =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status": "ok",
        "esp_id": esp_id,
        "detected_female_this_esp": detected_count,
        "total_detected_all_esp": total_all,
        "per_esp": ESP_RESULTS,
        "objects": filtered
    })

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        total = sum(v["count"] for v in ESP_RESULTS.values())
        return jsonify({
            "total_all_esp": total,
            "per_esp": ESP_RESULTS,
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)