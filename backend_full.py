from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# --- Chatbot ---
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
