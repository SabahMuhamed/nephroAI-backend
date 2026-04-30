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
import fitz  # PyMuPDF
import tempfile
import os
import re
import gc
from PIL import Image
import io


app = Flask(__name__)
CORS(app)

print("🚀 CKD API Running...")

# =========================
# LOAD MODELS
# =========================
rf_model = joblib.load("rf_model_egfr.pkl")
xgb_model = joblib.load("xgb_model_egfr.pkl")

columns = joblib.load("columns_egfr.pkl")
median = joblib.load("median.pkl")

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
# DROPDOWN NORMALIZATION
# =========================


def normalize_value(key, value):
    v = str(value).lower().strip()

    # Yes/No Fields (htn, dm, cad, pe, ane)
    if key in ["htn", "dm", "cad", "pe", "ane"]:
        return "yes" if "yes" in v or ("y" == v) else "no"

    # Normal/Abnormal (rbc, pc)
    if key in ["rbc", "pc"]:
        return "abnormal" if "abnormal" in v else "normal"

    # Presence (pcc, ba)
    if key in ["pcc", "ba"]:
        if "notpresent" in v or "not present" in v or "nil" in v or "absent" in v:
            return "notpresent"
        return "present" if "present" in v else "notpresent"

    # Appetite (appet)
    if key == "appet":
        return "poor" if "poor" in v else "good"

    return v

# =========================
# EXTRACTION LOGIC
# =========================


@app.route("/extract-report", methods=["POST"])
def extract_report():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        temp_path = tempfile.mktemp(suffix=".pdf")
        file.save(temp_path)

        text = ""
        # 1. Digital Extraction
        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except:
            pass

        # 2. OCR Fallback (Using fitz to avoid Poppler)
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

        # Regex for finding numbers
        def get_number(line):
            nums = re.findall(r"[\d.]+", line)
            return nums[0] if nums else None

        # --- DATA EXTRACTION LOOP ---
        for line in lines:
            # 1. Numeric Mappings (based on your report structure)
            if "age" in line:
                extracted["age"] = get_number(line)
            if "blood pressure" in line or "bp" in line:
                extracted["bp"] = get_number(line)
            if "specific gravity" in line:
                extracted["sg"] = get_number(line)
            if "albumin" in line:
                extracted["al"] = get_number(line)
            if "sugar" in line:
                extracted["su"] = get_number(line)
            if "blood glucose" in line:
                extracted["bgr"] = get_number(line)
            if "blood urea" in line:
                extracted["bu"] = get_number(line)
            if "creatinine" in line:
                extracted["sc"] = get_number(line)
            if "sodium" in line:
                extracted["sod"] = get_number(line)
            if "potassium" in line:
                extracted["pot"] = get_number(line)
            if "hemoglobin" in line:
                extracted["hemo"] = get_number(line)
            if "packed cell volume" in line:
                extracted["pcv"] = get_number(line)
            if "wbc" in line:
                extracted["wc"] = get_number(line)

            # Handle RBC (Number vs Status)
            if "rbc" in line:
                num = get_number(line)
                if num and float(num) > 3:
                    extracted["rc"] = num  # Numeric count
                else:
                    extracted["rbc"] = line  # Categorical status

            # 2. Categorical Mappings (Drop-down Fixes)
            if "pc" in line and "pus cell" not in line:
                extracted["pc"] = line
            if "pus cell" in line:
                extracted["pc"] = line
            if "pcc" in line or "clumps" in line:
                extracted["pcc"] = line
            if "ba" in line or "bacteria" in line:
                extracted["ba"] = line
            if "appet" in line or "appetite" in line:
                extracted["appet"] = line
            if "pe" in line or "edema" in line:
                extracted["pe"] = line
            if "ane" in line or "anemia" in line:
                extracted["ane"] = line
            if "htn" in line:
                extracted["htn"] = line
            if "dm" in line:
                extracted["dm"] = line
            if "cad" in line:
                extracted["cad"] = line

        # Global Conditions Scan
        if "hypertension" in text:
            extracted["htn"] = "yes"
        if "diabetes" in text:
            extracted["dm"] = "yes"
        if "coronary" in text:
            extracted["cad"] = "yes"

        # Final Normalization
        final_data = {k: normalize_value(k, v)
                      for k, v in extracted.items() if v}

        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

        return jsonify(final_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# PREDICT
# =========================


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        patient_name = data.get("patient_name", "Unknown")

        # CLEAN EMPTY
        for k in data:
            if data[k] == "":
                data[k] = np.nan

        # VALIDATE
        errors = validate_input(data)
        if errors:
            return jsonify({"error": errors}), 400

        # =========================
        # PREPROCESS
        # =========================
        df = pd.DataFrame([data])

        df = df.apply(lambda x: x.astype(str).str.strip().str.lower())

        df = df.replace({
            "yes": 1, "no": 0,
            "normal": 1, "abnormal": 0,
            "present": 1, "notpresent": 0,
            "good": 1, "poor": 0
        })

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(median)

        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        # =========================
        # MODEL
        # =========================
        rf_prob = float(rf_model.predict_proba(df)[0][1])
        xgb_prob = float(xgb_model.predict_proba(df)[0][1])

        final_prob = (0.7 * xgb_prob) + (0.3 * rf_prob)

        # ✅ FIXED FORMAT
        prediction = "ckd" if final_prob > 0.6 else "not_ckd"

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "patient_name": patient_name,
            "prediction": prediction,
            "confidence": round(final_prob * 100, 2),
            "risk_level": "high" if final_prob > 0.75 else "medium" if final_prob > 0.5 else "low",
            "inputs": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
