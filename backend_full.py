from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- CORS ---
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# --- LOAD CSV HOẶC DỮ LIỆU MẶC ĐỊNH ---
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
# 🚀 RUN SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
