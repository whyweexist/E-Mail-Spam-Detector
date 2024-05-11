import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

#This code is solely written by DHRUV TIWARI 

nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)

# PorterStemmer object initiate
ps = PorterStemmer()

def transform_text(text):
    # lower casing
    text = text.lower()
    # converting text into list of words
    text = nltk.word_tokenize(text)

    y = []
    # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stopwords/helping words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Normalization of word i.e converting words into their base form.
    for j in text:
        y.append(ps.stem(j))

    return " ".join(y)

tfidf = pd.read_pickle('models/vectorizer.pkl')
model = pd.read_pickle('models/model.pkl')

st.title(':rainbow[*SMS/Email Spam Detection*]')
st.markdown("-------------------")
st.markdown('##### Discover if your text messages are safe or sneaky! Try this Email Spam Detection now!')
st.markdown('###### This model can detect spam messages with an accuracy of 97%.')

st.markdown(" ")
user_input = st.text_input('Enter your text here')

if st.button("Predict"):
    if user_input[:] == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess user input
        transformed_txt = transform_text(user_input)
        converted_num = tfidf.transform([transformed_txt])
        result = model.predict(converted_num)[0]
        transformed_txt = transform_text(user_input)
        print(f"Transformed Text: {transformed_txt}")  # Debug line
        converted_num = tfidf.transform([transformed_txt])
        print(f"TF-IDF Vector: {converted_num.toarray()}")  # Debug line
        result = model.predict(converted_num)[0]
        print(f"Prediction: {result}")  # Debug line


        # Display prediction
        if result == 1:
            st.error("SPAM")
        else:
            st.success("Not Spam")
