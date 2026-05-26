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
GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# ================= PERSISTENT STORAGE =================
ESP_RESULTS_FILE = "esp_results.json"
ESP_RESULTS      = {}
ESP_LOCK         = Lock()

# ================= GITHUB CONFIG =================
GITHUB_REPO       = "onlykartika/ESP32-CAM"
GITHUB_FOLDER     = "images"
GITHUB_API_ROOT   = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_HEADERS    = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "User-Agent":    "Render-AI-Server",
    "Accept":        "application/vnd.github.v3+json"
}

# ================= LOAD / SAVE =================
def save_esp_results():
    try:
        with open(ESP_RESULTS_FILE, "w") as f:
            json.dump(ESP_RESULTS, f, indent=2)
        print("[INFO] Saved esp_results locally")
    except Exception as e:
        print(f"[ERROR] Save local failed: {e}")

def load_esp_results():
    global ESP_RESULTS
    if os.path.exists(ESP_RESULTS_FILE):
        try:
            with open(ESP_RESULTS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ESP_RESULTS = data
                    print("[INFO] Loaded esp_results from local file")
                    return
        except Exception as e:
            print(f"[ERROR] Failed load local: {e}")

    try:
        res = requests.get(
            f"{GITHUB_API_ROOT}/esp_results.json",
            headers=GITHUB_HEADERS, timeout=10
        )
        if res.status_code == 200:
            content    = base64.b64decode(res.json()["content"]).decode()
            data       = json.loads(content)
            if isinstance(data, dict):
                ESP_RESULTS = data
                save_esp_results()
                print("[INFO] Loaded esp_results from GitHub")
                return
    except Exception as e:
        print(f"[WARN] Failed load from GitHub: {e}")

    ESP_RESULTS = {}
    print("[INFO] ESP_RESULTS initialized empty")

load_esp_results()

# ================= ROBOFLOW CONFIG =================
# ⚠️ Tetap pakai konfigurasi asli — JANGAN diubah
WORKSPACE_NAME = "my-workspace-rrwxa"
WORKFLOW_ID    = "detect-count-and-visualize"
TARGET_LABEL   = "female"
CONF_THRESHOLD = 0.4

rf_client = None

def get_rf_client():
    global rf_client
    if rf_client is None:
        rf_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",  # tetap pakai detect
            api_key=ROBOFLOW_API_KEY
        )
    return rf_client

# ================= ROBOFLOW REST FALLBACK =================
def run_roboflow_rest(image_bytes):
    """Fallback: kirim langsung via REST tanpa SDK"""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    url       = f"https://serverless.roboflow.com/{WORKSPACE_NAME}/{WORKFLOW_ID}"
    payload   = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type":  "base64",
                "value": image_b64
            }
        }
    }
    resp = requests.post(url, json=payload, timeout=60)
    print(f"[DEBUG] REST status: {resp.status_code}")
    print(f"[DEBUG] REST response: {resp.text[:500]}")
    if resp.status_code != 200:
        raise Exception(f"REST error {resp.status_code}: {resp.text}")
    return resp.json()

# ================= PARSE PREDICTIONS =================
def parse_predictions(result):
    """
    Coba semua kemungkinan struktur output Roboflow.
    Logika sama persis dengan Colab yang berhasil.
    """
    predictions = []

    try:
        # Struktur list (workflow output)
        if isinstance(result, list) and result:
            for item in result:
                if not isinstance(item, dict):
                    continue
                if "predictions" in item:
                    raw = item["predictions"]
                    if isinstance(raw, list):
                        predictions.extend(raw)
                    elif isinstance(raw, dict) and "predictions" in raw:
                        predictions.extend(raw["predictions"])

        # Struktur dict
        elif isinstance(result, dict):
            if "predictions" in result:
                raw = result["predictions"]
                if isinstance(raw, list):
                    predictions = raw
                elif isinstance(raw, dict) and "predictions" in raw:
                    predictions = raw["predictions"]
            elif "result" in result and isinstance(result["result"], list):
                predictions = result["result"]

    except Exception as e:
        print(f"[WARN] Parse error: {e}")

    return predictions

# ================= HEALTH =================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "message":         "AI server running - Male Detection Ready",
        "target_label":    TARGET_LABEL,
        "conf_threshold":  CONF_THRESHOLD,
        "workspace":       WORKSPACE_NAME
    })

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    image_data = request.data
    if not image_data:
        return jsonify({"error": "no image data"}), 400

    image_size = len(image_data)
    print(f"[INFO] Received image: {image_size} bytes")

    if image_size < 1000:
        return jsonify({
            "error": f"gambar terlalu kecil ({image_size} bytes)",
            "hint":  "Thunder Client: tab Body > Binary > pilih file JPG"
        }), 400

    esp_id = request.headers.get("X-ESP-ID", "unknown")

    # Ambil timestamp dari ESP jika ada
    esp_timestamp = request.headers.get("X-Timestamp")
    try:
        timestamp = int(esp_timestamp) if esp_timestamp else int(time.time())
    except Exception:
        timestamp = int(time.time())

    filename = f"{esp_id}_{timestamp}.jpg"
    print(f"[INFO] Upload dari {esp_id} | size: {image_size} bytes")

    # Simpan ke disk
    try:
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"[INFO] Saved: {filename}")
    except Exception as e:
        return jsonify({"error": "save image failed", "detail": str(e)}), 500

    # ===== ROBOFLOW: SDK dulu, fallback REST =====
    result      = None
    method_used = ""

    try:
        rf     = get_rf_client()
        result = rf.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filename},
            use_cache=False
        )
        method_used = "SDK"
        print("[INFO] Roboflow SDK berhasil")
        try:
            print("[DEBUG] SDK result:\n" + json.dumps(result, indent=2, default=str))
        except Exception:
            print("[DEBUG] SDK result:", str(result)[:500])

    except Exception as sdk_err:
        print(f"[WARN] SDK gagal: {sdk_err} — mencoba REST...")
        try:
            result      = run_roboflow_rest(image_data)
            method_used = "REST"
            print("[INFO] REST berhasil")
        except Exception as rest_err:
            print(f"[ERROR] REST juga gagal: {rest_err}")
            if os.path.exists(filename):
                os.remove(filename)
            return jsonify({
                "error":      "roboflow failed",
                "sdk_error":  str(sdk_err),
                "rest_error": str(rest_err)
            }), 500

    # ===== UPLOAD GAMBAR KE GITHUB =====
    github_success = False
    try:
        with open(filename, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        folder_path = f"{GITHUB_FOLDER}/esp_{esp_id.split('_')[-1] if '_' in esp_id else esp_id}"
        put_url     = f"{GITHUB_API_ROOT}/{folder_path}/{filename}"

        get_res = requests.get(put_url, headers=GITHUB_HEADERS, timeout=5)
        sha     = get_res.json().get("sha") if get_res.status_code == 200 else None

        payload = {
            "message": f"upload from {esp_id} ({filename})",
            "content": img_b64
        }
        if sha:
            payload["sha"] = sha

        res = requests.put(put_url, headers=GITHUB_HEADERS, json=payload, timeout=15)
        if res.status_code in (200, 201):
            github_success = True
            print(f"[INFO] GitHub image: {res.status_code}")
        else:
            print(f"[WARN] GitHub image failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[WARN] GitHub image error: {e}")

    if os.path.exists(filename):
        os.remove(filename)

    # ===== PARSE PREDICTIONS =====
    predictions_list = parse_predictions(result)

    print(f"[DEBUG] method={method_used}, total predictions={len(predictions_list)}")
    for p in predictions_list:
        lbl  = p.get("class") or p.get("label") or "unknown"
        conf = float(p.get("confidence") or p.get("score") or 0.0)
        print(f"   → '{lbl}' ({conf*100:.1f}%)")

    # Filter target
    filtered = []
    for p in predictions_list:
        label = p.get("class") or p.get("label") or "unknown"
        conf  = float(p.get("confidence") or p.get("score") or 0.0)
        if label.lower() == TARGET_LABEL.lower() and conf >= CONF_THRESHOLD:
            filtered.append({
                "label":      label,
                "confidence": round(conf * 100, 2)
            })

    detected_count = len(filtered)
    print(f"[INFO] '{TARGET_LABEL}' terdeteksi: {detected_count}")

    # ===== UPDATE & SAVE =====
    with ESP_LOCK:
        ESP_RESULTS[esp_id] = {
            "count":       detected_count,
            "last_update": timestamp * 1000
        }
        save_esp_results()

        try:
            json_content = json.dumps(ESP_RESULTS, indent=2).encode()
            content_b64  = base64.b64encode(json_content).decode()

            get_res = requests.get(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS, timeout=10
            )
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None

            put_data = {
                "message": f"Update from {esp_id} - {time.strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64
            }
            if sha:
                put_data["sha"] = sha

            put_res = requests.put(
                f"{GITHUB_API_ROOT}/esp_results.json",
                headers=GITHUB_HEADERS, json=put_data, timeout=15
            )
            print(f"[INFO] GitHub JSON: {put_res.status_code}")
        except Exception as e:
            print(f"[WARN] GitHub JSON error: {e}")

    total_all = sum(v["count"] for v in ESP_RESULTS.values())

    return jsonify({
        "status":               "ok",
        "esp_id":               esp_id,
        "method_used":          method_used,
        "image_size_bytes":     image_size,
        "detected_this_esp":    detected_count,
        "total_all_esp":        total_all,
        "per_esp":              ESP_RESULTS,
        "objects":              filtered,
        "github_image_success": github_success
    })

# ================= SUMMARY =================
@app.route("/summary", methods=["GET"])
def summary():
    with ESP_LOCK:
        return jsonify({
            "total_all_esp": sum(v["count"] for v in ESP_RESULTS.values()),
            "per_esp":       ESP_RESULTS
        })

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
