# phobert_service.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ViolenceDetector:
    def __init__(self):
        print("🔄 Loading PhoBERT for violence detection...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "NlpHUST/vietnamese-violence-detection"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("✅ PhoBERT loaded!")

    def predict(self, text: str):
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**tokens).logits
            prob = torch.softmax(logits, dim=1)[0][1].item()  # index 1 = violent

        return prob > 0.5   # true = violent

# Tạo 1 instance duy nhất
violence_detector = ViolenceDetector()

def is_violent(text):
    return violence_detector.predict(text)
