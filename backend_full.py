from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Các import mới cho model ViT
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# --- Chatbot phần cũ ---
try:
    df = pd.read_csv("traindata.csv")
except FileNotFoundError:
    df = pd.DataFrame([
        {"question": "What is school violence?", "answer": "School violence refers to harmful behaviors."},
        {"question": "How to prevent school violence?", "answer": "By reporting bullying and creating safe environment."},
        {"question": "Who to talk to if bullied?", "answer": "Talk to a teacher or counselor."}
    ])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input.strip():
        return jsonify({"reply": "Please enter a message."})
    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, X)
    idx = sim.argmax()
    reply = df.iloc[idx]["answer"]
    return jsonify({"reply": reply})

@app.route("/train", methods=["POST"])
def train():
    new_data = request.json.get("data", [])
    if not new_data:
        return jsonify({"message": "No data provided"})
    global df, X, vectorizer
    df = pd.DataFrame(new_data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["question"])
    return jsonify({"message": "Model trained successfully"})

@app.route("/contact", methods=["POST"])
def contact():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")
    print(f"Contact from {name} ({email}): {message}")
    return jsonify({"message": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


# ==============================
# --- PHẦN VIOLENCE IMAGE ---
# ==============================
MODEL_NAME = "jaranohaal/vit-base-violence-detection"

# Load model & feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Preprocess
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    # Nhãn model: mình coi nhãn thứ 1 là violent, thứ 0 là non-violent (cần check model)
    prob_violent = probs[1].item()
    prob_nonviolent = probs[0].item()

    return jsonify({
        "prob_violent": prob_violent,
        "prob_nonviolent": prob_nonviolent
    })


# ==============================
# --- TÍNH TOÁN VIDEO BẰNG FRAME ---
# ==============================
@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400

    file = request.files["video"]
    filename = file.filename
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    import cv2
    cap = cv2.VideoCapture(save_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    frame_id = 0
    interval = int(fps)  # phân tích mỗi 1 second 1 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            # chuyển frame sang PIL Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = feature_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits
            p = torch.softmax(logits, dim=1)[0]
            results.append({
                "frame": frame_id,
                "prob_violent": p[1].item()
            })
        frame_id += 1
    cap.release()

    # Tìm frame nguy hiểm nhất
    max_frame = max(results, key=lambda x: x["prob_violent"]) if results else None
    return jsonify({
        "frames": results,
        "max_frame": max_frame
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
