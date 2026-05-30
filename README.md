

# Skin Cance<img width="1459" height="912" alt="Capture d&#39;écran 2026-05-30 200113" src="https://github.com/user-attachments/assets/abb77208-8f13-4900-bc72-b56c5b04553f" />
<img width="1114" height="880" alt="Capture d&#39;écran 2026-05-30 200155" src="https://github.com/user-attachments/assets/d0de1233-7178-4e60-a4a9-c3881343b3a9" />
<img width="1014" height="759" alt="Capture d&#39;écran 2026-05-30 200139" src="https://github.com/user-attachments/assets/2add6eee-a81c-467c-9cd7-0388fac97d36" />
<img width="1251" height="735" alt="Capture d&#39;écran 2026-05-30 200125" src="https://github.com/user-attachments/assets/f65d2eda-fee8-432a-848f-2a99faeca452" />

 Detection Web Application

## Overview

This project is a web-based application developed using Flask, TensorFlow, and OpenCV for automated skin cancer detection from dermoscopic images.

The application enables healthcare professionals or users to upload a skin lesion image and receive an AI-based prediction indicating whether the lesion is classified as Benign or Malignant. A detailed medical report can also be generated in PDF format.

---

## Objectives

The main objectives of this project are:

* Develop an intelligent skin cancer detection system.
* Integrate a deep learning model into a web application.
* Provide a simple and user-friendly interface.
* Generate professional medical reports automatically.

---

## Technologies Used

* Python
* Flask
* TensorFlow / Keras
* OpenCV
* NumPy
* ReportLab
* Bootstrap 5
* HTML / CSS

---

## Application Features

### Patient Information Form

The user enters:

* First Name
* Last Name
* Age
* Gender
* Body Part

### Image Upload

The application allows uploading a skin lesion image for analysis.

### AI Prediction

The deep learning model analyzes the uploaded image and predicts:

* Benign
* Malignant

### Confidence Score

The system provides a confidence percentage indicating the reliability of the prediction.

### Risk Assessment

Based on the prediction confidence, a risk level is displayed:

* Low Risk
* Medium Risk
* High Risk

### PDF Medical Report

A professional PDF report can be generated containing:

* Patient information
* Diagnosis result
* Confidence score
* Risk level
* Date of analysis

---

## Project Structure

SKIN_CANCER_APP/

* app.py
* model/

  * vgg16_skin_cancer.h5
* static/

  * uploads/
  * medical_report.pdf
* templates/

  * predict.html
  * result.html
  * login.html
  * dashboard.html
  * history.html

---

## How to Run the Application

1. Install the required Python packages.
2. Place the trained model file inside the model folder.
3. Run the application using:

python app.py

4. Open your browser and navigate to:

http://127.0.0.1:5000

---

## Future Improvements

Possible future enhancements include:

* Patient history management
* Database integration
* Multiple disease classification
* User authentication system
* Cloud deployment
* Advanced medical analytics dashboard

---

## Author

Nour Trabelsi

Final Year Engineering Project

2026
