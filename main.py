import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN


word_index=imdb.get_word_index()
reversed_word_index={value:key for key,value in word_index.items()}

model=load_model('SimpleRNNIMDB.h5')

maxlen = 500

def  decode_review(enocded_review):
    return ' '.join([reversed_word_index.get(i-3,'?') for i in enocded_review])
def pre_processing_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review




## Streamlit app
st.title('Movie Review Sentiment Analysis')
st.write('This is a sentiment analysis app for movie reviews. You can enter your review and get the sentiment analysis.')
st.write('Enter your review:')

user_review=st.text_area('Enter your review:')

if st.button('Classify'):
    preprosessinput=pre_processing_text(user_review)
    prediction=model.predict(preprosessinput)
    sentiment='Positive' if prediction[0]>0.5 else 'Negative'

    st.write('Sentiment:',sentiment)
    st.write('Score:',prediction[0][0]) 
else:
    st.write('Enter your review:')
    




