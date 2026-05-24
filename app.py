import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center;'>📩 SMS Spam Detection</h1>", unsafe_allow_html=True)
st.write("")

# Input box
input_sms = st.text_area("✉️ Enter your message here:")

# Button
if st.button("🔍 Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = vectorizer.transform([input_sms])
        result = model.predict(transformed_sms)[0]

        if result == 1:
            st.error(f"🚨 Spam Message\n\nConfidence:")
        else:
            st.success(f"✅ Not Spam (Ham)\n\nConfidence:")