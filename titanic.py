import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

st.set_option('deprecation.showPyplotGlobalUse',False)


titanic=pd.read_csv("C:/Users/Pratik/OneDrive/Desktop/javascript/train (1).csv")
df=pd.DataFrame(titanic)
# st.write(df.head())

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
#st.write(df.head())

inputs = df.drop('Survived',axis='columns')
target = df.Survived


dummies = pd.get_dummies(inputs.Sex)

inputs = pd.concat([inputs,dummies],axis='columns')

inputs.drop(['Sex','male'],axis='columns',inplace=True)

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(X_train,y_train)



st.title('Titanic Classifer') # creates a title with name "SPAM HAM CLASSIFIER"
ip0 = st.number_input("pclass") #creates a text box , where we can enter text
ip1=st.number_input("Age")
ip2=st.number_input("Fare")
ip3=st.number_input("Gender")
ip=[ip0,ip1,ip2,ip3]
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])

