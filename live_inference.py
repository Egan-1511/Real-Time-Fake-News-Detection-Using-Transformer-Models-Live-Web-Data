import requests
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer
model_path = "./bert-fake-news"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Setup NewsAPI
API_KEY = '8bcb4e9a20f74aaa8208d3f910f1f4c8'
URL = f'https://newsapi.org/v2/top-headlines?language=en&pageSize=5&apiKey={API_KEY}'

def get_headlines():
    response = requests.get(URL)
    data = response.json()
    headlines = [article['title'] for article in data['articles']]
    return headlines

def predict_fake_real(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    label = "Fake" if pred.item() == 1 else "Real"
    return label, confidence.item()

if __name__ == "__main__":
    headlines = get_headlines()
    for headline in headlines:
        label, confidence = predict_fake_real(headline)
        print(f"Headline: {headline}")
        print(f"Prediction: {label} ({confidence*100:.2f}%)\n")
