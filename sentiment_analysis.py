import streamlit as st
import pickle
import re
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="AI Echo – Sentiment Analyzer")

# LOAD DATASET

def load_data():
    return pd.read_csv("reviews.csv")   

df = load_data()

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

menu = st.sidebar.selectbox(
    "Sentiment Analysis System",
    ["Sentiment Analyzer", "Key Questions"]
)


if menu == "Sentiment Analyzer":
    st.header("AI Echo – Review Sentiment Analysis")
    
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
        if prediction == "4":
            st.success("🙂 Positive review detected")
        elif prediction == "2":
            st.info("😐 Neutral review detected")
        else:
            st.error("☹️ Negative review detected")
        
if menu == "Key Questions":
    st.header("Review Sentiment Analysis")
    
    
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["review"].astype(str).apply(classify_sentiment)
df["sentiment_score"] = df["review"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["review_length"] = df["review"].astype(str).apply(len)

# ---------------------------------------------------
# Sidebar Questions
# ---------------------------------------------------
st.sidebar.header("📌 Choose a Key Question")

questions = [
    "1. Overall sentiment of user reviews",
    "2. Sentiment variation by rating",
    "3. Keywords/phrases associated with each sentiment",
    "4. Sentiment trend over time",
    "5. Verified vs non-verified sentiment",
    "6. Review length vs sentiment",
    "7. Sentiment by location",
    "8. Sentiment by platform",
    "9. Sentiment across ChatGPT versions",
    "10. Most common negative feedback themes"
]

choice = st.sidebar.selectbox("Select question", questions)

st.header(f"🔍 {choice}")

# ---------------------------------------------------
# 1. Overall Sentiment Proportion
# ---------------------------------------------------
if choice == questions[0]:
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_percent = sentiment_counts / len(df) * 100

    st.subheader("Sentiment Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Counts")
        st.write(sentiment_counts)

    with col2:
        st.write("### Percentages")
        st.write(sentiment_percent.round(2))

    # Charts
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%")
    ax2.set_title("Sentiment Proportion")
    st.pyplot(fig2)

# ---------------------------------------------------
# 2. Sentiment Variation by Rating
# ---------------------------------------------------
elif choice == questions[1]:
    st.write("### Rating vs Sentiment")
    pivot = pd.crosstab(df["rating"], df["sentiment"])
    st.write(pivot)

    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment by Rating")
    st.pyplot(fig)

# ---------------------------------------------------
# 3. Word Clouds per Sentiment
# ---------------------------------------------------
elif choice == questions[2]:
    st.write("### Word Clouds by Sentiment")

    for label in ["Positive", "Neutral", "Negative"]:
        st.write(f"#### {label} Reviews")
        text = " ".join(df[df["sentiment"] == label]["review"].astype(str))
        wc = WordCloud(width=800, height=300).generate(text)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------------------------------------
# 4. Sentiment Trend Over Time
# ---------------------------------------------------
elif choice == questions[3]:
    if "date" not in df.columns:
        st.error("❌ Column 'date' not found. Add a date column to enable trend analysis.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        trend = df.groupby(df["date"].dt.to_period("M"))["sentiment_score"].mean()

        st.write("### Sentiment Over Time")
        fig, ax = plt.subplots()
        trend.plot(ax=ax)
        ax.set_title("Monthly Sentiment Trend")
        st.pyplot(fig)

# ---------------------------------------------------
# 5. Verified vs Non-Verified Sentiment
# ---------------------------------------------------
elif choice == questions[4]:
    if "verified_purchase" not in df.columns:
        st.error("❌ Column 'verified_purchase' missing.")
    else:
        st.write("### Sentiment by Verified Purchase")
        pivot = pd.crosstab(df["verified_purchase"], df["sentiment"])
        st.write(pivot)

        fig, ax = plt.subplots()
        pivot.plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------
# 6. Review Length vs Sentiment
# ---------------------------------------------------
elif choice == questions[5]:
    st.write("### Relationship Between Review Length & Sentiment Score")

    fig, ax = plt.subplots()
    ax.scatter(df["review_length"], df["sentiment_score"], alpha=0.5)
    ax.set_xlabel("Review Length")
    ax.set_ylabel("Sentiment Score")
    st.pyplot(fig)

# ---------------------------------------------------
# 7. Sentiment by Location
# ---------------------------------------------------
elif choice == questions[6]:
    if "location" not in df.columns:
        st.error("❌ Column 'location' missing.")
    else:
        st.write("### Sentiment by Location")
        pivot = pd.crosstab(df["location"], df["sentiment"])
        st.write(pivot)

# ---------------------------------------------------
# 8. Sentiment by Platform
# ---------------------------------------------------
elif choice == questions[7]:
    if "platform" not in df.columns:
        st.error("❌ Column 'platform' missing.")
    else:
        st.write("### Sentiment by Platform")
        pivot = pd.crosstab(df["platform"], df["sentiment"])
        st.write(pivot)

# ---------------------------------------------------
# 9. ChatGPT Version vs Sentiment
# ---------------------------------------------------
elif choice == questions[8]:
    if "version" not in df.columns:
        st.error("❌ Column 'version' missing.")
    else:
        st.write("### Sentiment by ChatGPT Version")
        pivot = pd.crosstab(df["version"], df["sentiment"])
        st.write(pivot)

        fig, ax = plt.subplots()
        pivot.plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------
# 10. Negative Review Themes
# ---------------------------------------------------
elif choice == questions[9]:
    st.write("### Keywords in Negative Reviews")

    negative_text = " ".join(df[df["sentiment"] == "Negative"]["review"].astype(str))
    wc = WordCloud(width=900, height=400).generate(negative_text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
