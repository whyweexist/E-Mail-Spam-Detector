import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model

tfidf = pd.read_pickle('./models/tfidf.pickle')
model = load_model('./models/model.weights.best.hdf5')

def prediction(text):
    pred = model.predict(text)
    return pred

def pre_process(text):
    return tfidf.transform([text]).toarray()
   
def get_class(value):
    if value == 0:
        return 'FRAUD'
    elif value == 1:
        return 'NORMAL'
    else:
        return 'SPAM'

def main():
   st.title(':rainbow[*SMS/Email Spam Detection*]')
   st.markdown("-------------------")
   st.markdown('##### Discover if your text messages are safe or sneaky! Try this Email Spam Detection now!')
   st.markdown('###### This model can detect spam messages with an accuracy of 97%.')

   st.markdown(" ")
   text_input = st.text_area('Enter your text here')


   

   #st.title("Spam or Fraud Message Prediction")
   #st.write("This app is created to predict if a email message is Spam, Fraud or Normal")
   #text_input = st.text_area('Enter some text')
    result = None
    value = None

    vec = pre_process(text_input)
    if st.button("Predict"):
         if text_input[:] == "":
                st.warning("Please enter a message.")
    else:
        value = np.argmax(prediction(vec))
        result = get_class(value) 
        st.subheader('Prediction')
        st.markdown(f'The predicted message is: **{result}**' )
   

if __name__=='__main__':
    main()