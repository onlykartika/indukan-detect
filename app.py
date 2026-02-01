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
GITHUB_API_IMAGES = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FOLDER}"
GITHUB_API_ROOT = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent": "Render-AI-Server",
    "Accept": "application/vnd.github.v3+json"
}

# ================= LOAD / SAVE RESULTS =================
def load_esp_results():
    global ESP_RESULTS

    # 1. Load local
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                ESP_RESULTS = json.load(f)
                print("[INFO] Loaded esp_results from local file")
                return
        except Exception as e:
            print(f"[ERROR] Failed load local: {e}")

    # 2. Fallback GitHub
    try:
        res = requests.get(
            f"{GITHUB_API_ROOT}/esp_results.json",
            headers=GITHUB_HEADERS,
            timeout=10
        )
        if res.status_code == 200:
            content = base64.b64decode(res.json()["content"]).decode()
            ESP_RESULTS = json.loads(content)
            save_esp_results()
            print("[INFO] Loaded esp_results from GitHub")
            return
    except Exception as e:
        print(f"[WARN] Failed load from GitHub: {e}")

    ESP_RESULTS = {}

def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)
        print("[INFO] Saved esp_results to local")
    except Exception as e:
        print(f"[ERROR] Save local failed: {e}")

load_esp_results()

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
TARGET_LABEL = "male"
CONF_THRESHOLD = 0.6

# ================= HEALTH =================
@app.route("/", methods=["GET"])
def health():
    return "AI server running (Female Detection) - Ready!"

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    if not request.data:
        return jsonify({"error": "no image"}), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")
    timestamp = int(time.time())
    filename = f"{esp_id}_{timestamp}.jpg"

    print(f"[INFO] Upload dari {esp_id}, size: {len(request.data)} bytes")

    # Save sementara
    try:
        with open(filename, "wb") as f:
            f.write(request.data)
    except Exception as e:
        return jsonify({"error": "save image failed", "detail": str(e)}), 500

    # ===== ROBOFLOW WORKFLOW =====
    try:
        rf = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
        print("[INFO] Roboflow workflow selesai")
    except Exception as e:
        os.remove(filename)
        return jsonify({"error": "roboflow failed", "detail": str(e)}), 500

    # ===== UPLOAD IMAGE TO GITHUB =====
    try:
        with open(filename, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        put_url = f"{GITHUB_API_IMAGES}/{esp_id}/{filename}"
        res = requests.put(
            put_url,
            headers=GITHUB_HEADERS,
            json={
                "message": f"Upload dari {esp_id}: {filename}",
                "content": content_b64
            },
            timeout=15
        )

        if res.status_code in (200, 201):
            print("[INFO] Image uploaded to GitHub")
        else:
            print(f"[WARN] GitHub upload gagal: {res.status_code}")
    except Exception as e:
        print(f"[WARN] GitHub upload error: {e}")

    if os.path.exists(filename):
        os.remove(filename)

    # ===== PARSE PREDICTIONS =====
    predictions = []
    if isinstance(result, dict) and "predictions" in result:
        predictions = result["predictions"]
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "predictions" in item:
                predictions.extend(item["predictions"])

    filtered = []
    for p in predictions:
        if not isinstance(p, dict):
            continue
        label = p.get("class") or p.get("label")
        conf = p.get("confidence") or p.get("score") or 0

        if label and label.lower() == TARGET_LABEL.lower() and conf >= CONF_THRESHOLD:
            filtered.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

    detected_count = len(filtered)

    # ===== UPDATE RESULTS & BACKUP =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count": detected_count,
            "last_update": int(time.time() * 1000)
        }
        save_esp_results()

        try:
            json_content = json.dumps(ESP_RESULTS).encode()
            content_b64 = base64.b64encode(json_content).decode()

            get_res = requests.get(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS,
                timeout=10
            )
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            put_data = {
                "message": f"Update dari {esp_id} - {time.strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64
            }
            if sha:
                put_data["sha"] = sha

            requests.put(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS,
                json=put_data,
                timeout=15
            )
        except Exception as e:
            print(f"[WARN] Backup JSON error: {e}")

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
        return jsonify({
            "total_all_esp": sum(v["count"] for v in ESP_RESULTS.values()),
            "per_esp": ESP_RESULTS
        })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
