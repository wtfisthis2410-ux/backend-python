from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://frontend-1-b0bm.onrender.com"]}})

# ====================================================
# 1. LOAD TOKENIZER + BASE MODEL + ADAPTER
# ====================================================
BASE_MODEL_NAME = "vinai/phobert-base"
ADAPTER_PATH = "./adapter_model"  # thư mục chứa adapter_model.safetensors + adapter_config.json

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=6)

print("Loading PEFT adapter...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="auto")
peft_model.eval()

# ====================================================
# 2. BOT RESPONSES MAPPING
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

LABEL_MAPPING = {
    0: "greeting",
    1: "normal",
    2: "violence",
    3: "complain",
    4: "ask_help",
    5: "end"
}

DEFAULT_RESPONSE = "Mimir chưa hiểu ý bạn lắm, bạn có thể nói lại được không?"

# ====================================================
# 3. CHAT ENDPOINT
# ====================================================
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Bạn chưa nhập tin nhắn nào cả."})

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(peft_model.device)
    with torch.no_grad():
        outputs = peft_model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=-1))
        predicted_label = LABEL_MAPPING.get(pred_id, "normal")

    reply = random.choice(RESPONSES.get(predicted_label, [DEFAULT_RESPONSE]))
    return jsonify({"reply": reply})

# ====================================================
# 4. TRAINING ENDPOINT (chỉ cập nhật dữ liệu CSV)
# ====================================================
@app.route("/train", methods=["POST"])
def train():
    new_data = request.json.get("data", [])
    if not new_data:
        return jsonify({"message": "No data provided"})

    # Lưu CSV, adapter fine-tune vẫn giữ nguyên
    import pandas as pd
    df = pd.DataFrame(new_data)
    df.to_csv("train_data.csv", index=False)
    return jsonify({"message": "Data saved. Adapter model not retrained automatically."})

# ====================================================
# 5. CONTACT + HEALTH CHECK
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
