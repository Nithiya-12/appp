import numpy as np
import pandas as pd
import re
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Sample dataset (you can replace with CSV)
# ----------------------------
data = {
    "text": [
        "Win money now",
        "Limited time offer",
        "Free prize waiting",
        "Call me tomorrow",
        "Let's meet for lunch",
        "Project meeting today"
    ],
    "label": [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# ----------------------------
# Text cleaning (NLP)
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# ----------------------------
# Tokenization
# ----------------------------
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
max_len = 50
X = pad_sequences(sequences, maxlen=max_len)
y = df['label']

# Save tokenizer
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

# ----------------------------
# Deep Learning Model (LSTM)
# ----------------------------
model = Sequential()
model.add(Embedding(5000, 64, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=5, verbose=1)

# Save model
model.save("spam_model.h5")

print("Model trained and saved!")