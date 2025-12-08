from flask import Flask, request, jsonify
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
from io import BytesIO
import os
import cv2

app = Flask(__name__)

# --- Load model ViT ---
MODEL_NAME = "jaranohaal/vit-base-violence-detection"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)


# ============================
#      DETECT IMAGE (file)
# ============================
@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    prob_violent = probs[1].item()
    prob_nonviolent = probs[0].item()

    return jsonify({
        "prob_violent": prob_violent,
        "prob_nonviolent": prob_nonviolent,
        "violent": prob_violent > 0.5  # ⭐ THÊM CHO GIỐNG FRONTEND
    })


# ============================
#      DETECT VIDEO (file)
# ============================
@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    filename = file.filename

    # save file
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    # read video
    cap = cv2.VideoCapture(save_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps), 1)

    total_frames = 0
    violent_frames = 0

    frame_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            total_frames += 1

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = feature_extractor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)[0]
            violent_prob = probs[1].item()

            if violent_prob > 0.5:
                violent_frames += 1

            results.append({
                "frame": frame_id,
                "prob_violent": violent_prob
            })

        frame_id += 1

    cap.release()

    violent_rate = violent_frames / total_frames if total_frames > 0 else 0

    return jsonify({
        "frames": results,
        "violent_rate": violent_rate,     # ⭐ KHỚP VỚI FRONTEND
        "violent": violent_rate > 0.3     # ⭐ FRONTEND ĐỌC "violent"
    })


# ============================
#        RUN SERVER
# ============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
