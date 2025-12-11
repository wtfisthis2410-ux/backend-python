from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# ====================================================
# 1. LOAD TRAIN DATA
# ====================================================
try:
    df = pd.read_csv("train_data.csv")
except FileNotFoundError:
    df = pd.DataFrame([
        {"text": "Chào bạn", "label": "greeting"},
        {"text": "Mình bị đánh", "label": "violence"},
        {"text": "Cảm ơn bạn", "label": "end"}
    ])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# ====================================================
# 2. BOT RESPONSES
# ====================================================
RESPONSES = {
    "greeting": [
        "Chào bạn! Mimir đang nghe bạn đây.",
        "Xin chào! Bạn muốn chia sẻ điều gì?",
        "Hello, mình ở đây để trò chuyện với bạn!"
    ],
    "normal": [
        "Mình hiểu rồi! Bạn muốn nói thêm gì không?",
        "Cảm ơn bạn đã chia sẻ, mọi thứ vẫn ổn chứ?",
        "Nghe có vẻ là một ngày bình thường đó."
    ],
    "violence": [
        "Mình rất tiếc khi nghe điều đó. Bạn có ổn không?",
        "Chuyện đó nghiêm trọng đấy… bạn có thể kể chi tiết hơn không?",
        "Nếu bạn thấy bất an, hãy nói với thầy cô hoặc người lớn mà bạn tin tưởng nhé."
    ],
    "complain": [
        "Mình nghe nè… điều đó chắc khiến bạn mệt mỏi lắm.",
        "Ai cũng có những ngày tệ… bạn muốn tâm sự thêm không?",
        "Nghe có vẻ bạn đã chịu áp lực khá nhiều."
    ],
    "ask_help": [
        "Bạn cần giúp gì? Mimir luôn sẵn sàng hỗ trợ.",
        "Được thôi, bạn đang cần trợ giúp ở phần nào?",
        "Bạn muốn mình hỗ trợ điều gì?"
    ],
    "end": [
        "Cảm ơn bạn đã chia sẻ! Khi nào cần cứ nhắn Mimir nhé.",
        "Chúc bạn một ngày tốt lành!",
        "Mimir luôn sẵn sàng khi bạn cần."
    ]
}

DEFAULT_RESPONSE = "Mimir chưa hiểu ý bạn lắm, bạn có thể nói lại được không?"

# ====================================================
# 3. CHAT ENDPOINT
# ====================================================
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if not user_input.strip():
        return jsonify({"reply": "Bạn chưa nhập tin nhắn nào cả."})

    vector = vectorizer.transform([user_input])
    predicted_label = model.predict(vector)[0]

    reply = random.choice(RESPONSES.get(predicted_label, [DEFAULT_RESPONSE]))

    return jsonify({"reply": reply})

# ====================================================
# 4. TRAIN ENDPOINT
# ====================================================
@app.route("/train", methods=["POST"])
def train():
    new_data = request.json.get("data", [])

    if not new_data:
        return jsonify({"message": "No data provided"})

    global df, X, y, vectorizer, model

    df = pd.DataFrame(new_data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression()
    model.fit(X, y)

    df.to_csv("train_data.csv", index=False)

    return jsonify({"message": "Model trained successfully"})

# ====================================================
# 5. CONTACT + HEALTH
# ====================================================
@app.route("/contact", methods=["POST"])
def contact():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    print(f"[CONTACT] From {name} ({email}): {message}")

    return jsonify({"message": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

# ====================================================
# 6. RUN SERVER
# ====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
