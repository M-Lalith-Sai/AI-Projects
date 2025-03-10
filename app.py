import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_analysis_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function for sentiment prediction
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)
    return "Positive" if prediction > 0.5 else "Negative"

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a sentence:")
if st.button("Analyze"):
    result = predict_sentiment(user_input)
    st.write(f"Sentiment: **{result}**")
