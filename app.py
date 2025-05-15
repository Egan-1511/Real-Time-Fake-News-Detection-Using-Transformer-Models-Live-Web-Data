from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer once at startup
model_path = "./bert-fake-news"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('headline', '')
    if not text:
        return jsonify({'error': 'No headline provided'}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)
    label = "Fake" if pred.item() == 1 else "Real"

    return jsonify({
        'label': label,
        'confidence': confidence.item()
    })

if __name__ == '__main__':
    app.run(debug=True)
