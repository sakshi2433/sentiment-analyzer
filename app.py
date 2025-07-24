import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


st.title("✈️ Twitter Airline Sentiment Analyzer")

tweet = st.text_area("Paste a tweet here:")

if st.button("Analyze Sentiment"):
    if tweet:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.subheader(f"🧠 Sentiment: **{prediction.upper()}**")
    else:
        st.warning("Please enter some text.")
