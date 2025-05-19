from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer once
model_path = "model/"  # Change this if your model is stored elsewhere
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_news(text):
    """Predicts if a news article is real (1) or fake (0)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return prediction, confidence
