import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index= imdb.get_word_index()

model=load_model('rnn_model.imdb.h5')
def preprocess_input(input_text):
    words = input_text.lower().split()
    encoded_text = [word_index.get(word, 2) +3 for word in words]
    padded_sequences = sequence.pad_sequences([encoded_text], maxlen=500)
    return padded_sequences

def predict_sentiment(input_text):
    preprocessed_input = preprocess_input(input_text)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

import streamlit as st
st.title("Sentiment Analysis with RNN") 
st.write("Enter a movie review its sentiment as positive or negative.")

user_input = st.text_input("Enter your text here:")
if st.button("CLASSIFY"):

    prediction=model.predict(preprocess_input(user_input))
    sentiment= 'positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f"The sentiment of the review is: {sentiment} with a confidence of {prediction[0][0]:.2f}")


else:
 st.write("Please enter some text to analyze.")