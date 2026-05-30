from datetime import datetime
import os

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, session
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


# =========================
# APP INIT
# =========================
app = Flask(__name__)
app.secret_key = "secret123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_PATH = os.path.join(BASE_DIR, "static", "medical_report.pdf")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
target_path = os.path.join(BASE_DIR, "model", "vgg16_skin_cancer.h5")


def ensure_upload_folder():
    if os.path.exists(UPLOAD_FOLDER) and not os.path.isdir(UPLOAD_FOLDER):
        raise RuntimeError(
            f"Upload path exists but is not a directory: {UPLOAD_FOLDER}"
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def build_unique_filepath(filename):
    name, ext = os.path.splitext(filename)
    candidate = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    counter = 1

    while os.path.exists(candidate):
        candidate = os.path.join(
            app.config["UPLOAD_FOLDER"], f"{name}_{counter}{ext}"
        )
        counter += 1

    return candidate


ensure_upload_folder()

# =========================
# LOAD MODEL
# =========================
model = load_model(os.path.join(BASE_DIR, "model", "vgg16_skin_cancer.h5"))
print("Model loaded successfully!")


# =========================
# PREDICTION FUNCTION
# =========================
def predict_skin_cancer(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return "Error (invalid image)", 0.0, "Low"

    try:
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = float(model.predict(img, verbose=0)[0][0])
    except Exception as exc:
        print("Prediction error:", exc)
        return "Error", 0.0, "Low"

    if prediction >= 0.5:
        label = "Malignant"
        confidence = prediction * 100
    else:
        label = "Benign"
        confidence = (1 - prediction) * 100

    confidence = round(confidence, 2)

    if label == "Malignant":
        if confidence >= 90:
            risk = "High"
        elif confidence >= 70:
            risk = "Medium"
        else:
            risk = "Low"
    else:
        risk = "Low"

    return label, confidence, risk


@app.route("/")
def home():
    return render_template("predict.html")


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()
    body_part = request.form.get("body_part", "").strip()

    if not file or file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename)
    if not filename:
        return "Invalid file name", 400

    ensure_upload_folder()
    filepath = build_unique_filepath(filename)
    file.save(filepath)

    saved_filename = os.path.basename(filepath)
    label, confidence, risk = predict_skin_cancer(filepath)

    session["last_result"] = {
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": gender,
        "body_part": body_part,
        "result": label,
        "confidence": confidence,
        "risk": risk,
        "image": saved_filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return render_template(
        "result.html",
        first_name=first_name,
        last_name=last_name,
        age=age,
        gender=gender,
        body_part=body_part,
        result=label,
        confidence=confidence,
        risk=risk,
        image=saved_filename,
    )


@app.route("/download_pdf")
def download_pdf():
    data = session.get("last_result")

    doc = SimpleDocTemplate(REPORT_PATH, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Skin Cancer Report", styles["Title"]))
    content.append(Spacer(1, 12))

    if not data:
        content.append(Paragraph("No prediction available.", styles["Normal"]))
    else:
        content.append(
            Paragraph(
                f"Generated on: {data.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
                styles["Normal"],
            )
        )
        content.append(Spacer(1, 8))
        content.append(
            Paragraph(
                f"Patient: {data['first_name']} {data['last_name']}", styles["Normal"]
            )
        )
        content.append(Paragraph(f"Age: {data['age']}", styles["Normal"]))
        content.append(Paragraph(f"Gender: {data['gender']}", styles["Normal"]))
        content.append(Paragraph(f"Body Part: {data['body_part']}", styles["Normal"]))
        content.append(Spacer(1, 10))
        content.append(Paragraph(f"Diagnosis: {data['result']}", styles["Heading2"]))
        content.append(
            Paragraph(f"Confidence: {data['confidence']}%", styles["Normal"])
        )
        content.append(Paragraph(f"Risk: {data['risk']}", styles["Normal"]))

    doc.build(content)
    return send_file(REPORT_PATH, as_attachment=True)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
