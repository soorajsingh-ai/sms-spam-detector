import streamlit as st
import pickle
import re

# load model & vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    return text

# UI
st.title("📩 AI Spam Detector")

msg = st.text_area("Enter message below to check spam here")

if st.button("Predict"):
    if msg.strip() != "":
        # clean text
        clean_msg = clean_text(msg)

        # vectorize
        vector = vectorizer.transform([clean_msg])

        # predict
        result = model.predict(vector)[0]

        # output
        if result == 1:
            st.error("🚨 This is SPAM message")
        else:
            st.success("✅ This is HAM (Normal) message")
    else:
        st.warning("⚠️ Please enter a message first")