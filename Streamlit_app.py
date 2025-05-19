import streamlit as st
from app.scraper import fetch_latest_news  
from app.predictor import predict_news  


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Real-Time Fake News Detector")
st.markdown("Predict whether a news headline is **Real** or **Fake** using a Transformer-based NLP model.")


st.subheader("🔍 Enter News Headline or Article")
user_input = st.text_area("Paste your news content or headline here", height=150)


if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a headline or news content.")
    else:
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_news(user_input)
        st.success(f"Prediction: {'🟢 Real' if prediction == 1 else '🔴 Fake'}")
        st.write(f"Confidence: {confidence*100:.2f}%")

st.markdown("---")


st.subheader("🌐 Check Live News Headlines")
if st.button("Fetch Live News"):
    with st.spinner("Fetching live news..."):
        headlines = fetch_latest_news()
    if not headlines:
        st.error("Failed to fetch news. Please check your scraper.")
    else:
        for i, article in enumerate(headlines[:5], 1):
            st.markdown(f"**{i}. {article['title']}**")
            pred, conf = predict_news(article['title'])
            label = "🟢 Real" if pred == 1 else "🔴 Fake"
            st.markdown(f"> Prediction: **{label}** ({conf*100:.1f}%)")
            st.markdown("---")


st.markdown("Made with ❤️ by Egan")
