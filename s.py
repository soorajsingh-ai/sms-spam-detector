import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
"""df1=pd.read_csv("spam_ham_india.csv")
df2=pd.read_csv("spam_dataset.csv")
print(df1.head())
print(df2.head())

#Column name Rename
df2.rename(columns={'message' : 'Msg' ,'label' : 'Label'},inplace=True)
df =pd.concat([df1,df2],ignore_index=True)

#Save 
df.to_csv("Final_dataset.csv" , index=False)
print("Combind Successfully")"""

df =pd.read_csv("Final_dataset.csv")
print(df.head())
print(df.tail())
print(df.columns)
(df.shape)
print(df.info())
print(df['Label'].value_counts())

#NUll Each column Check
print(df.isnull().sum())

#Total NUll Check in entire dataset
print(df.isnull().sum().sum())
print(df[df.isnull().any(axis=1)])

#drop null row
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)

#Data Preprocessing 

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return " ".join(words)
df['Msg'] = df['Msg'].apply(transform_text)
print(df.head())

#Text to Number

cv = CountVectorizer(max_features=3000)

#Before Split
X_train, X_test, y_train, y_test = train_test_split(
    df['Msg'], df['Label'].map({'ham':0,'spam':1}),
    test_size=0.2, random_state=2
)

# Vectorization
cv = CountVectorizer(max_features=3000, ngram_range=(1,2))

X_train = cv.fit_transform(X_train)
X_test= cv.transform(X_test)

#TF-IDF

"""tfidf = TfidfVectorizer(max_features=3000 , ngram_range=(1,2))

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)"""

print(X_train.shape)
print(X_test.shape)

#Model Train 
model =MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

#Second model Train
#model = LogisticRegression(C=2.0, max_iter=1000)
#model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)
print("Accuracy" , accuracy_score (y_test, y_pred))
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)

import pickle

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))