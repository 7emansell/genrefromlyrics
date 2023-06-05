#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time
import os
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix,classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import spacy


# In[3]:


rawdf = pd.read_csv('train.csv')
df = rawdf.dropna()
df


# In[4]:


#Dropping genres, non-English lyrics
excessgenres = ['Folk', 'Indie', 'Other']

df = df[df.Genre.isin(excessgenres) == False]
df = df[df.Language == 'en']


# In[5]:


#Taking random 900 songs from each genre
genres = ['Rock', 'Pop', 'Hip-Hop', 'Metal', 'Country', 'Jazz', 'Electronic', 'R&B']
rand = pd.DataFrame()
for genre in genres:
    newdf = df[df.Genre == genre]
    sample = newdf.sample(n = 900)
    rand = rand.append(sample)
rand   


# In[6]:


#Create lemmatized corpus and clean lyrics
nlp = spacy.load('en_core_web_sm')
lemmatizedcorpus = []
def clean(text):
    # Make lower
    text = text.lower()
    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")+ ['solo', 'im', 'youre']
    text_filtered = [word for word in text if not word in useless_words]
    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    #Lemmatize and add to corpus
    text_joined = nlp(' '.join(text_filtered))
    text_stemmed = [y.lemma_ for y in text_joined]
    lemmatizedcorpus.append(str(nlp(' '.join(text_stemmed))))

    return ' '.join(text_stemmed)

rand['Cleaned Lyrics'] = rand.apply(lambda row: clean(row.Lyrics), axis = 1)


# In[7]:


#Creating TFIDF matrix
tfidf = TfidfVectorizer(stop_words='english', lowercase=False)    
X = tfidf.fit_transform(lemmatizedcorpus)

tfidf_tokens = tfidf.get_feature_names_out()


matrix = pd.DataFrame(
    data=X.toarray(), 
    index=rand.iterrows(), 
    columns=tfidf_tokens
)


# In[8]:


matrix


# In[16]:


#Processing for models
matrix['target'] = LabelEncoder().fit_transform(rand["Genre"])
target = matrix['target']

X_train,X_test,y_train,y_test = train_test_split(X, target, test_size=0.2, stratify = target)


# In[34]:


# Gradient Boosting Classifier

gbmodel = GradientBoostingClassifier(n_estimators=500, random_state=123)
gbmodel.fit(X_train, y_train)
print("Gradient Boosting accuracy score {:.2f} %\n".format(gbmodel.score(X_test,y_test)*100))


# In[35]:


# Random Forest Classifier

rfmodel = RandomForestClassifier()
rfmodel.fit(X_train, y_train)
print("Random forest accuracy score {:.2f} %\n".format(rfmodel.score(X_test,y_test)*100))


# In[40]:


# Naive Bayes Classifier

nbmodel = MultinomialNB()
nbmodel.fit(X_train.toarray(), y_train)
print("Naive Bayes accuracy score {:.2f} %\n".format(nbmodel.score(X_test.toarray(),y_test)*100))


# In[42]:


# Grid search to improve model performance (Gradient Boost)

parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.1, 0.2],
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "n_estimators":[10]
    }

gcv = GridSearchCV(gbmodel,parameters)
gcv.fit(X_train, y_train.values.ravel())
print(gcv.best_params_)


# In[43]:


# Grid search to improve model performance (Random Forest)

parameters = {
    'n_estimators': [5,50,100],
    'max_depth': [2,10,20,None]
}

rcv = GridSearchCV(rfmodel,parameters)
rcv.fit(X_train, y_train.values.ravel())
print(rcv.best_params_)


# In[45]:


# Grid search to improve model performance (Naive Bayes)

parameters ={'alpha': [0.00001, 0.001, 0.1, 1, 10, 100]}

ncv = GridSearchCV(nbmodel,parameters)
ncv.fit(X_train, y_train.values.ravel())
print(ncv.best_params_)


# In[51]:


# Optimized Gradient Boosting Classifier

ogbmodel = GradientBoostingClassifier(learning_rate = 0.01, max_depth= 8, max_features= 'sqrt', n_estimators = 10)
ogbmodel.fit(X_train, y_train)
print("Gradient Boosting accuracy score {:.2f} %\n".format(ogbmodel.score(X_test,y_test)*100))


# In[52]:


# Optimized Random Forest Classifier

orfmodel = RandomForestClassifier(max_depth= 20, n_estimators= 100)
orfmodel.fit(X_train, y_train)
print("Random forest accuracy score {:.2f} %\n".format(orfmodel.score(X_test,y_test)*100))


# In[53]:


# Optimized Naive Bayes Classifier

onbmodel = MultinomialNB(alpha=0.1)
onbmodel.fit(X_train.toarray(), y_train)
print("Naive Bayes accuracy score {:.2f} %\n".format(onbmodel.score(X_test.toarray(),y_test)*100))


# In[9]:


#Graph results
import matplotlib.pyplot as plt

graph = pd.DataFrame(
    {'Name': ['Multinomial NB', 'Grid Search Multinomial NB', 'Gradient Booster', 
              'Grid Search Gradient Booster', 'Random Forest', 'Grid Search Random Forest'], 
     'Accuracy': [nbmodel.score(X_test.toarray(),y_test), onbmodel.score(X_test.toarray(),y_test), gbmodel.score(X_test,y_test), ogbmodel.score(X_test,y_test), rfmodel.score(X_test,y_test), orfmodel.score(X_test,y_test)]}
)

plt.bar("Name", "Accuracy", data = graph, color=["blue", "green","blue", "green","blue", "green"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation = 90)
plt.axhline(y= 0.125, linewidth=1, color="k")
plt.title("Top 3 models final performance")
plt.show()






