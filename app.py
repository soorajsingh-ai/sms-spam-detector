import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

# Load model (ONLY ONCE)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center;'>📩 SMS Spam Detection</h1>", unsafe_allow_html=True)
st.write("")

# Input
input_sms = st.text_area("📩 Enter your message here:")

# Button
if st.button("Predict"):

    # Transform
    transformed_sms = vectorizer.transform([input_sms])

    # Predict
    result = model.predict(transformed_sms)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Normal Message (Ham)")