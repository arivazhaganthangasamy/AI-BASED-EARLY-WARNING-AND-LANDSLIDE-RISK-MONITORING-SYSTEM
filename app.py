# -*- coding: utf-8 -*-
"""AI-BASED EARLY WARNING AND LANDSLIDE RISK MONITORING SYSTEM (Flask)
Run:  pip install flask joblib pandas numpy scikit-learn requests
      python app.py     ->  http://127.0.0.1:5000
Telegram: set env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
"""
import os, io, wave, datetime as dt
import joblib, numpy as np, pandas as pd, requests
from flask import Flask, render_template, request, jsonify, Response, send_file
from dotenv import load_dotenv
load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "Landslide_Risk_Optimized_RandomForest_Model.pkl")
LOG_FILE = os.path.join(BASE, "landslide_prediction_log.csv")
IMAGE_DIR = os.path.join(BASE, "uploaded_site_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

app = Flask(__name__)
package = joblib.load(MODEL_FILE)
model, FEATURES = package["model"], package["features"]
TH = package.get("risk_thresholds", {"low": 0.20, "medium": 0.45, "high": 0.70})
RISK_ORDER = ["Low", "Medium", "High", "Critical"]

def risk_level(p):
    return "Low" if p < TH["low"] else "Medium" if p < TH["medium"] else "High" if p < TH["high"] else "Critical"

def predict(values):
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p = float(proba[classes.index(1)] if 1 in classes else proba[-1])
    return p, risk_level(p)

# ---------- alarm sounds ----------
def _wav(sig, sr):
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return buf.getvalue()

_cache = {}
def make_alarm(loud):
    if loud in _cache: return _cache[loud]
    sr = 22050
    if not loud:
        d = 0.35; t = np.linspace(0, d, int(sr * d), endpoint=False)
        beep = 0.25 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, np.minimum(t, d - t) / 0.02)
        sig = np.concatenate([np.concatenate([beep, np.zeros(int(sr * .25))]) for _ in range(3)])
    else:
        t = np.linspace(0, 8, sr * 8, endpoint=False)
        ph = 2 * np.pi * np.cumsum(1050 + 450 * np.sin(2 * np.pi * 1.5 * t)) / sr
        sig = 0.6 * np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
        sig = sig / np.max(np.abs(sig)) * 0.98
    _cache[loud] = _wav(sig, sr)
    return _cache[loud]

# ---------- telegram ----------
def build_message(level, p, v, location):
    head = {"Low": "🟢 LANDSLIDE UPDATE: LOW RISK", "Medium": "🟡 LANDSLIDE ADVISORY: MEDIUM RISK",
            "High": "🟠 LANDSLIDE WARNING: HIGH RISK", "Critical": "🔴 CRITICAL LANDSLIDE ALERT: TAKE ACTION NOW"}[level]
    act = {"Low": "Normal monitoring.", "Medium": "Stay alert and monitor conditions.",
           "High": "Prepare to evacuate. Avoid slopes and roads in the area.",
           "Critical": "Evacuate the danger zone immediately and inform authorities."}[level]
    return (f"{head}\n\n📍 Location: {location}\n📊 Probability: {p*100:.1f}%\n"
            f"🌧️ Rainfall: {v['Rainfall_Intensity_mm']} mm\n💧 Soil Moisture: {v['Soil_Moisture_percent']}%\n"
            f"📐 Slope Angle: {v['Slope_Angle_degree']}°\n📈 Ground Movement: {v['Ground_Movement_mm_per_day']} mm/day\n"
            f"🌡️ Temperature: {v['Temperature_C']}°C\n💦 Groundwater Level: {v['Groundwater_Level_m']} m\n"
            f"📜 Historical Landslide: {'Yes' if v['Historical_Landslide'] == 1 else 'No'}\n\n"
            f"⚠️ Action: {act}\n\n🕒 Time: {dt.datetime.now():%d-%b-%Y %H:%M}")

def send_telegram(body):
    token = "8527986741:AAHkfmkKD6-tsjDoZOTPRPYyrVsvemRhYb0"   # your bot token from BotFather
    chat = "5511579171"                               # your chat ID
    if not token or not chat:
        return False, "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": body}, timeout=30)
        if r.status_code == 200 and r.json().get("ok"):
            return True, "Telegram alert sent successfully."
        return False, f"Telegram failed. HTTP {r.status_code}: {r.text}"
    except requests.exceptions.Timeout:
        return False, "Telegram request timed out."
    except Exception as e:
        return False, f"Telegram failed: {e}"

# ---------- log ----------
def append_log(values, p, level, location, tg, image):
    row = {"Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Location": location, **values,
           "Probability": p, "Risk_Level": level, "Telegram_Sent": tg, "Image": image}
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

def load_log():
    return pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()

# ---------- pages ----------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/about")
def about(): return render_template("about.html")

@app.route("/predict")
def predict_page(): return render_template("predict.html")

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html")

# ---------- APIs ----------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    f = request.form
    try:
        values = {k: float(f[k]) for k in FEATURES if k != "Historical_Landslide"}
        values["Historical_Landslide"] = 1 if f.get("Historical_Landslide") in ("1", "Yes") else 0
    except (KeyError, ValueError):
        return jsonify(error="Please fill in all seven readings with valid numbers."), 400
    location = f.get("location", "Unknown site")
    p, level = predict(values)

    image_name = ""
    img = request.files.get("image")
    if img and img.filename:
        image_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + os.path.basename(img.filename)
        img.save(os.path.join(IMAGE_DIR, image_name))

    tg, tg_msg = "Skipped", ""
    if f.get("telegram") == "1":
        ok, tg_msg = send_telegram(build_message(level, p, values, location))
        tg = "Yes" if ok else "Failed"
    append_log(values, p, level, location, tg, image_name)
    return jsonify(probability=p, level=level, telegram=tg, telegram_msg=tg_msg,
                   loud=level in ("High", "Critical"), thresholds=TH)

@app.route("/alarm/<kind>.wav")
def alarm(kind): return Response(make_alarm(kind == "loud"), mimetype="audio/wav")

@app.route("/api/dashboard")
def api_dashboard():
    log = load_log()
    perf = {k: package.get(k, 0) * 100 for k in
            ("test_accuracy", "test_precision", "test_recall", "test_f1_score", "best_cv_accuracy")}
    fi = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1])
    out = dict(perf=perf, features=[a for a, _ in fi], importance=[float(b) for _, b in fi],
               thresholds=TH, counts={r: 0 for r in RISK_ORDER}, rows=[], total=0)
    if not log.empty:
        out["total"] = len(log)
        out["counts"].update({k: int(v) for k, v in log["Risk_Level"].value_counts().items()})
        out["rows"] = log.tail(200).fillna("").to_dict("records")
    return jsonify(out)

@app.route("/download/log")
def download_log():
    if not os.path.exists(LOG_FILE): return "No log yet.", 404
    return send_file(LOG_FILE, as_attachment=True, download_name="landslide_prediction_log.csv")

if __name__ == "__main__":
    app.run(debug=True)
