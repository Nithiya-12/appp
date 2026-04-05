import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# Sample dataset
# ----------------------------
data = {
    "text": [
        "Win money now",
        "Free lottery offer",
        "Claim your prize",
        "Hello how are you",
        "Let's meet tomorrow",
        "Project discussion today"
    ],
    "label": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# ----------------------------
# Vectorization (NLP)
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# ----------------------------
# Model (Naive Bayes)
# ----------------------------
model = MultinomialNB()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")
