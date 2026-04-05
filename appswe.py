import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load model & tokenizer
# ----------------------------
model = tf.keras.models.load_model("spam_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_len = 50

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# ----------------------------
# Prediction
# ----------------------------
def predict_spam(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    result = model.predict(padded)[0][0]
    return result

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 AI Spam Detection System (NLP + Deep Learning)")

msg = st.text_area("Enter Email Message")

# REAL-TIME detection
if msg:
    score = predict_spam(msg)

    st.write(f"Spam Probability: {score:.2f}")

    if score > 0.5:
        st.error("🚨 Spam Message Detected")
    else:
        st.success("✅ Safe Message")