import os
import cv2
import numpy as np
import pickle
import mysql.connector
import datetime
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# ✅ Load trained SVM model and label encoder
with open("svm_classifier.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Connect to MySQL Database on Railway
def connect_db():
    try:
        db_url = os.getenv("MYSQL_URL")  # Fetch Railway's MySQL URL from environment variables
        if not db_url:
            raise ValueError("Database URL not found. Make sure the Railway environment variable is set.")
        
        conn = mysql.connector.connect(
            option_files=db_url  # Use Railway's MySQL URL directly
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# ✅ Function to mark attendance in MySQL
def mark_attendance(name):
    conn = connect_db()
    if conn is None:
        return

    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (%s, %s)", (name, now))
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ Attendance marked for {name} at {now}")

# ✅ API Endpoint for Face Recognition
@app.route("/recognize", methods=["POST"])
def recognize_face():
    try:
        file = request.files["image"]  # Get the uploaded image
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Extract FaceNet embedding
        print("🔍 Extracting face embedding...")
        embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        embedding = np.array(embedding).reshape(1, -1)  # Reshape for SVM

        # Predict using SVM
        prediction = svm_model.predict(embedding)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        print(f"🎉 Recognized as: {predicted_label}")
        mark_attendance(predicted_label)

        return jsonify({"status": "success", "name": predicted_label})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
