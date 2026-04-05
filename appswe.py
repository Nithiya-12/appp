import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load model & tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("spam_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tokenizer

model, tokenizer = load_model()

max_len = 50

# ----------------------------
# Clean text (NLP)
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# ----------------------------
# Prediction function
# ----------------------------
def predict_spam(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    prediction = model.predict(padded)

    return prediction[0][0]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 AI Spam Detection System")

st.write("Enter your email message below 👇")

msg = st.text_area("Message")

# REAL-TIME OUTPUT
if msg:
    score = predict_spam(msg)

    st.write(f"Spam Probability: {score:.2f}")

    if score > 0.5:
        st.error("🚨 This is a SPAM message")
    else:
        st.success("✅ This is NOT a spam message")
