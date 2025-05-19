# 📰 Real-Time Fake News Detection Using Transformer Models and Live Web Data

This project focuses on detecting fake news in real time by leveraging advanced Natural Language Processing (NLP) techniques using transformer models. It combines machine learning, web scraping, and real-time inference to classify news articles as fake or real.

## 🚀 Features

- Real-time news scraping from online sources
- Preprocessing and vectorization of news headlines and content
- Fine-tuned Transformer model (e.g., BERT) for fake news classification
- Live dashboard to monitor predictions
- Continuous model improvement using user feedback (optional)

## 📊 Dataset Source

We used the **Fake and Real News Dataset** available on Kaggle:

📦 [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- **fake.csv** - Fake news articles
- **true.csv** - Real news articles

### Preprocessing Steps:
- Removed stopwords and punctuation
- Lowercased all text
- Used `nltk` for tokenization and basic cleaning
- Applied label encoding (Fake = 0, Real = 1)

## 🧠 Model Architecture

We fine-tuned a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model using the Hugging Face `transformers` library.

### Tools & Libraries:
- Python
- Hugging Face Transformers
- Scikit-learn
- NLTK
- BeautifulSoup & Requests (for web scraping)
- Streamlit / Flask (for real-time UI)
- Pandas / NumPy
- Matplotlib / Seaborn (for visualization)

## 🔍 Real-Time Component

We scraped articles from sources like:
- [News API](https://newsapi.org/)
- [Google News](https://news.google.com/)
- [Inshorts](https://www.inshorts.com/en/read)

These articles were preprocessed and passed through the model pipeline in real-time to predict their authenticity.

## 📈 Model Performance

| Metric     | Score |
|------------|-------|
| Accuracy   | ~96%  |
| Precision  | ~95%  |
| Recall     | ~96%  |
| F1-Score   | ~95%  |

Confusion matrix and ROC-AUC visualizations are included in the notebook.

## 🛠️ Project Structure

      real-time-fake-news-detection/
│
├── data/
│ ├── fake.csv
│ └── true.csv
│
├── notebooks/
│ └── training_and_evaluation.ipynb
│
├── model/
│ └── best_model.bin
│
├── app/
│ ├── scraper.py
│ ├── predictor.py
│ └── streamlit_app.py
│
├── requirements.txt
└── README.md


## 🧪 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/real-time-fake-news-detection.git
cd real-time-fake-news-detection

2. Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Start the real-time app:

bash
Copy
Edit
streamlit run app/streamlit_app.py

✨ Future Improvements
Improve scraping from more diverse and international sources

Implement multi-language support

Add user feedback loop for retraining

Deploy on cloud (e.g., AWS/GCP/Azure)

👨‍💻 Author
Project by Egan

