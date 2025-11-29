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

@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    prob_violent = probs[1].item()
    prob_nonviolent = probs[0].item()

    return jsonify({
        "prob_violent": prob_violent,
        "prob_nonviolent": prob_nonviolent
    })

@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400

    file = request.files["video"]
    filename = file.filename
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    cap = cv2.VideoCapture(save_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    frame_id = 0
    interval = max(int(fps), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
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

    max_frame = max(results, key=lambda x: x["prob_violent"]) if results else None
    return jsonify({
        "frames": results,
        "max_frame": max_frame
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
