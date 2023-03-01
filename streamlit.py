from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

rad=st.sidebar.radio("Navigation",["Home","Sentiment Analysis","Sarcasm Detection"])

#Home Page
if rad=="Home":
    st.title("Complete Text Analysis App")
    st.image("ss.jpeg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")


#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)





#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page
if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text!!")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("Negetive Text!!")
        elif prediction2==1:
            st.success("Positive Text!!")


#Sarcasm Detection Prediction
tfidf5=TfidfVectorizer(stop_words=sw,max_features=20)
def transform5(txt1):
    txt2=tfidf5.fit_transform(txt1)
    return txt2.toarray()

df5=pd.read_csv("Sarcasm Detection.csv")
df5.columns=["Text","Label"]
x=transform5(df5["Text"])
y=df5["Label"]
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.1,random_state=0)
model5=LogisticRegression()
model5.fit(x_train5,y_train5) 

#Sarcasm Detection Page
if rad=="Sarcasm Detection":
    st.header("Detect Whether The Text Is Sarcastic Or Not!!")
    sent5=st.text_area("Enter The Text")
    transformed_sent5=transform_text(sent5)
    vector_sent5=tfidf5.transform([transformed_sent5])
    prediction5=model5.predict(vector_sent5)[0]

    if st.button("Predict"):
        if prediction5==1:
            st.exception("Sarcastic Text!!")
        elif prediction5==0:
            st.success("Non Sarcastic Text!!")
