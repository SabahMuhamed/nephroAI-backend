# =========================
# IMPORTS
# =========================
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pdfplumber
import pytesseract
import fitz
import tempfile
import os
import re
import gc
from PIL import Image
import io
from xgboost import XGBClassifier

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
CORS(app)

print("🚀 CKD API Running...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD MODELS
# =========================
print("Loading models...")

rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model_egfr.pkl"))

xgb_model = XGBClassifier()
xgb_model.load_model(os.path.join(BASE_DIR, "xgb_model_egfr.json"))

columns = joblib.load(os.path.join(BASE_DIR, "columns_egfr.pkl"))
median = joblib.load(os.path.join(BASE_DIR, "median.pkl"))

print("Models loaded successfully")

# =========================
# VALIDATION
# =========================


def validate_input(data):
    errors = []
    required_fields = [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
        "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"{field} missing")

    return errors

# =========================
# NORMALIZATION
# =========================


def normalize_value(key, value):
    v = str(value).lower().strip()

    if key in ["htn", "dm", "cad", "pe", "ane"]:
        return "yes" if "yes" in v or v == "y" else "no"

    if key in ["rbc", "pc"]:
        return "abnormal" if "abnormal" in v else "normal"

    if key in ["pcc", "ba"]:
        if "notpresent" in v or "not present" in v or "nil" in v:
            return "notpresent"
        return "present"

    if key == "appet":
        return "poor" if "poor" in v else "good"

    return v

# =========================
# EXTRACT REPORT
# =========================


@app.route("/extract-report", methods=["POST"])
def extract_report():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        file.save(temp_path)

        text = ""

        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except:
            pass

        if len(text.strip()) < 100:
            doc = fitz.open(temp_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img) + "\n"
            doc.close()

        text = text.lower()
        lines = text.split("\n")
        extracted = {}

        def get_number(line):
            nums = re.findall(r"[\d.]+", line)
            return nums[0] if nums else None

        for line in lines:
            if "age" in line:
                extracted["age"] = get_number(line)
            if "blood pressure" in line:
                extracted["bp"] = get_number(line)
            if "specific gravity" in line:
                extracted["sg"] = get_number(line)
            if "albumin" in line:
                extracted["al"] = get_number(line)
            if "sugar" in line:
                extracted["su"] = get_number(line)
            if "glucose" in line:
                extracted["bgr"] = get_number(line)
            if "urea" in line:
                extracted["bu"] = get_number(line)
            if "creatinine" in line:
                extracted["sc"] = get_number(line)

        final_data = {k: normalize_value(k, v)
                      for k, v in extracted.items() if v}

        os.remove(temp_path)
        gc.collect()

        return jsonify(final_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# HOME
# =========================


@app.route("/")
def home():
    return "CKD API running"

# =========================
# PREDICT
# =========================


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        patient_name = data.get("patient_name", "Unknown")

        for k in data:
            if data[k] == "":
                data[k] = np.nan

        errors = validate_input(data)
        if errors:
            return jsonify({"error": errors}), 400

        df = pd.DataFrame([data])

        df = df.apply(lambda x: x.astype(str).str.lower().str.strip())

        df = df.replace({
            "yes": 1, "no": 0,
            "normal": 1, "abnormal": 0,
            "present": 1, "notpresent": 0,
            "good": 1, "poor": 0
        })

        df = df.apply(pd.to_numeric, errors="coerce")

        # ✅ FIXED median usage
        df = df.fillna(pd.Series(median))

        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        rf_prob = float(rf_model.predict_proba(df)[0][1])
        xgb_prob = float(xgb_model.predict_proba(df)[0][1])

        final_prob = (0.7 * xgb_prob) + (0.3 * rf_prob)

        prediction = "ckd" if final_prob > 0.6 else "not_ckd"

        return jsonify({
            "patient_name": patient_name,
            "prediction": prediction,
            "confidence": round(final_prob * 100, 2),
            "risk_level": "high" if final_prob > 0.75 else "medium" if final_prob > 0.5 else "low"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
