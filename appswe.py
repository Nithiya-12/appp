import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model safely
model = tf.keras.models.load_model("spam_model.h5", compile=False)
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_len = 50

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def predict_spam(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    result = float(model.predict(padded, verbose=0)[0][0])
    return result

st.title("📩 AI Spam Detection System")

msg = st.text_area("Enter Email Message")

if msg:
    score = predict_spam(msg)

    st.write(f"Spam Probability: {score:.2f}")

    if score > 0.5:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
