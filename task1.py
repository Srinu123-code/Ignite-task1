import pandas as pd
import seaborn as s 
import numpy as np
import matplotlib.pyplot as plt
import nltk
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\Dell\\Desktop\\Reviews.csv")
data = pd.DataFrame(data)

data1= data.head(300)
data1 =pd.DataFrame(data1)
#data['ProfileName'].fillna('a', inplace=True)
#data['Summary'].fillna('okay', inplace=True)

porter_stemmer = PorterStemmer()

def stemmer(d):
    cleaned_data = re.sub('[^a-zA-Z]',' ',d)
    cleaned_data= cleaned_data.lower()
    cleaned_data = cleaned_data.split()
    cleaned_data = [porter_stemmer.stem(word) for word in cleaned_data if not word in stopwords.words('english')]
    cleaned_data = ' '.join(cleaned_data)
    
    return cleaned_data
    
data1['cleandata']=data1['Text'].apply(stemmer) 

"""

def polarized_data(d):
    polarity1=TextBlob(d)
    if polarity1.sentiment[0]>0:
        return 'positive'
    if polarity1.sentiment[0]<0:
        return 'negative'
    else:
        return 'neutral' 
"""    
def polarized_data(d):
    polarity1=TextBlob(d)
    if polarity1.polarity>0:
        return 'positive'
    if polarity1.polarity<0:
        return 'negative'
    else:
        return 'neutral'


data1['polarized_data']=data1['cleandata'].apply(polarized_data) 
print(data1['polarized_data'].value_counts())   

# model creation

x= data1['cleandata'].values
y=data1['polarized_data'].values

X_train, X_test, y_train, y_test = train_test_split(x,y ,test_size = 0.6)

# we use TfidVectorizer for convertinf text data to numerical data
pattern =TfidfVectorizer()

X_train = pattern.fit_transform(X_train)
X_test = pattern.transform(X_test)

model = LogisticRegression()

model.fit(X_train,y_train)
pred=model.predict(X_train)

acc= accuracy_score(y_train,pred)
print(acc)

     

pred_test=model.predict(X_test)  
  
acc_test= accuracy_score(y_test, pred_test)   
print(acc_test)

import pickle

f_name = 'sentimental_prediction_model.sav'
pickle.dump(model, open(f_name,'wb'))

