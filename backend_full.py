from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORT THÊM CHO XỬ LÝ ẢNH / VIDEO ---
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CORS ---
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# --- LOAD CSV HOẶC DÙNG DỮ LIỆU MẶC ĐỊNH ---
try:
    df = pd.read_csv("traindata.csv")
except FileNotFoundError:
    df = pd.DataFrame([
        {"question": "What is school violence?", "answer": "School violence refers to harmful behaviors that can occur among students or between students and teachers."},
        {"question": "How can we prevent school violence?", "answer": "We can prevent school violence by promoting kindness, reporting bullying, and creating a safe school environment."},
        {"question": "Who should I talk to if I'm being bullied?", "answer": "You should talk to a teacher, counselor, or a trusted adult immediately."}
    ])

# --- TF-IDF ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

# ===============================
# 🧠 CHATBOT API
# ===============================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input.strip():
            return jsonify({"reply": "Please enter a message."})

        user_vec = vectorizer.transform([user_input])
        sim = cosine_similarity(user_vec, X)
        idx = sim.argmax()
        reply = df.iloc[idx]["answer"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


# ===============================
# 🏋️ TRAIN API
# ===============================
@app.route("/train", methods=["POST"])
def train():
    global df, X, vectorizer
    new_data = request.json.get("data", [])
    if not new_data:
        return jsonify({"message": "No data provided"})

    df = pd.DataFrame(new_data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["question"])
    return jsonify({"message": "Model trained successfully"})


# ===============================
# 📩 CONTACT FORM API
# ===============================
@app.route("/contact", methods=["POST"])
def contact():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    print(f"📩 Contact message from {name} ({email}): {message}")
    return jsonify({"message": "Message received successfully!"})


# ===============================
# ❤️ HEALTH CHECK
# ===============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend is running"})


# ===============================
# 🔥 HÀM PHÂN TÍCH BẠO LỰC (GIẢ LẬP)
# ===============================
def detect_violence(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Ngưỡng demo: dao động mạnh = có thể có bạo lực
    if blur > 200:
        return True, blur
    return False, blur


# ===============================
# 🖼️ API NHẬN DIỆN ẢNH
# ===============================
@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)

    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    img = cv2.imread(file_path)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    is_violent, score = detect_violence(img)

    return jsonify({
        "violent": is_violent,
        "score": float(score)
    })


# ===============================
# 🎥 API NHẬN DIỆN VIDEO
# ===============================
@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)

    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot read video"}), 400

    violent_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        is_violent, _ = detect_violence(frame)
        if is_violent:
            violent_frames += 1

    cap.release()

    ratio = violent_frames / max(total_frames, 1)

    return jsonify({
        "total_frames": total_frames,
        "violent_frames": violent_frames,
        "violence_ratio": ratio,
        "violent": ratio > 0.1
    })


# ===============================
# 🚀 RUN SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
