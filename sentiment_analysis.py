import streamlit as st
import pickle
import re

# LOAD MODEL + TFIDF
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))


# TEXT CLEAN FUNCTION

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# STREAMLIT UI

st.set_page_config(page_title="AI Echo – Sentiment Analyzer")

st.title("AI Echo – Review Sentiment Analysis")
st.write("Analyze sentiment of ChatGPT App Reviews using Logistic Regression.")

review_input = st.text_area("Enter a review:", height=150)

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("Please enter a review text!")
    else:
        cleaned = clean_text(review_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.subheader("Prediction Result:")
        if prediction == "positive":
            st.success("🙂 Positive review detected")
        elif prediction == "neutral":
            st.info("😐 Neutral review detected")
        else:
            st.error("☹️ Negative review detected")