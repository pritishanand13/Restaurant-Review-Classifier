import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 
from sklearn.naive_bayes import MultinomialNB


# creating a data frame to read data from dataset
df = pd.read_table('Restaurant_Reviews.tsv')
#creating two variables for review and liked 
x = df['Review'].values
y = df['Liked'].values

#using train_test_split to train model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

vect = CountVectorizer(stop_words = 'english')
x_train_vect = vect.fit_transform(x_train) 
x_test_vect = vect.transform(x_test)

#Method 1
#Apply the ML algorithm (Support vector machine)(SVM)
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train_vect,y_train)
y_pred1 = model1.predict(x_test_vect)
# calculating accuracy score for above model created by Support Vectoriser
acc1 = accuracy_score(y_pred1,y_test)

#Method 2
# SVC pipeline method 
# combines two estimators (countvect+svc)
model2 = make_pipeline(CountVectorizer(),SVC())
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
acc2 = accuracy_score(y_pred2,y_test)

#Method 3
#Using Naive Bayes
model3 =MultinomialNB()
model3.fit(x_train_vect,y_train)
y_pred3 = model3.predict(x_test_vect)
acc3 = accuracy_score(y_pred3,y_test)

# method 4 using pipeline(countvect,multinomialNB)  
model4 = make_pipeline(CountVectorizer(),MultinomialNB())
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
accuracy_score(y_pred4,y_test)

# ACCURACY FOR SVC - 77%
# SVC PIPELINE - 80.5 % 
# ACCURACY FOR MultinomialNB - 79.5%
# MultinomialNB PIPELINE - 84 %

# using joblib to save pipeline model with highest accuracy
joblib.dump(model4,'Restaurant Review')
reload_model = joblib.load('Restaurant Review')

reload_model = joblib.load('Restaurant Review')  

st.title("POSITIVE AND NEGATIVE REVIEW CLASSIFICATION")
#st.write("USING PIPELINE MODELS")
input = st.text_input("Enter your message:") 

#predict if the entered review is positive or negative

output = reload_model.predict([input])  
if st.button('PREDICT'): 
  if output[0] == 1:  st.title("Positive")
  else: st.title("Negative")
