import pandas as pd
import re
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import svm

from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('fake news data.csv')

df.sample(10)

df.shape

df.describe()

df.info()

df.describe(include=['O'])

df.columns

df['Label'].value_counts().plot(kind = "barh")

sns.distplot(df['Label'])

 df.isnull().sum()

X = df.drop('URLs', axis=1)
y = df['URLs']

X = df['Headline']
y = df['Label']
X
y

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

print("Training feature set size:",X_train.shape)
print("Test feature set size:",X_test.shape)
print("Training variable set size:",y_train.shape)
print("Test variable set size:",y_test.shape)

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(tfidf_train, y_train)

y_pred = svm_classifier.predict(tfidf_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(X_train)
tfid_x_test = tfvect.transform(X_test)
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)

def fake_news_data(Headline):
    input_data = [Headline]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)

#test cases
fake_news_data ('9/28 Through the 40s: The Gloaming; HBD Bill, Lou, Cy the Third, Everett, Cy, Leon & Buck; Clarke Honored;Tiny 2-Hitter')

fake_news_data('Shanghai Masters: Kyle Edmund beats Jiri Vesely to reach second round')

fake_news_data('The ‘Sweet Season’ in Coming Up, But with Good Habits and Careful Monitoring, Teeth Problems can be Avoided, According to Gilroy Dentist')









