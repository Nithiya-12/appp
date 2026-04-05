import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Prediction
def predict_spam(text):
    vec = vectorizer.transform([text])
    result = model.predict_proba(vec)[0][1]
    return result

# UI
st.title("📩 Spam Detection (ML + NLP)")

msg = st.text_area("Enter message")

# Real-time
if msg:
    score = predict_spam(msg)

    st.write(f"Spam Probability: {score:.2f}")

    if score > 0.5:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
