#!/usr/bin/env python
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
from math import*
import nltk
from gensim.models.keyedvectors import KeyedVectors
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True ,limit=500000)


def square_rooted(x):
 	return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 	numerator = sum(a*b for a,b in zip(x,y))
   	denominator = square_rooted(x)*square_rooted(y)
   	return round(numerator/float(denominator),3)

def tokenize(text):
    return([text.split(',', 1)[0].strip()])

# read the data in
df = pd.read_csv("comp_1.csv", delimiter = ',')
#print df

#print df.isnull().sum().sum()
user_df = pd.read_csv("users.csv", delimiter = ',')
#print(len(user_df))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = tokenize,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 
user_vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = tokenize,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 

Keywords_cat = vectorizer.fit_transform(df.Keywords)
train_data_features = Keywords_cat.toarray()
Skills_cat = user_vectorizer.fit_transform(user_df.Skills)
user_train_data_features = Skills_cat.toarray()

#print cosine_similarity(user_train_data_features,train_data_features)
#print len(user_train_data_features)

vocab = vectorizer.get_feature_names()
#print vocab
user_vocab = user_vectorizer.get_feature_names()
ranking = []
arr = []
for i in range(len(user_df)):
    user_skills = [x.strip() for x in user_df['Skills'][i].split(',')]
    #print user_skills
    for j in range(len(df)):
        ans = 0
        skills = [x.strip() for x in df['Keywords'][j].split(',')]
        for k in range(len(user_skills)):
            for t in range(len(skills)):
                try:
                    ans = ans + model.similarity(skills[t],user_skills[k])
                    #print model.similarity(skills[t],user_skills[k])
                except:
                    try:
                        expanded1 = [x.strip() for x in skills[t].split()]
                        #print expanded1 
                        expanded2 = [x.strip() for x in user_skills[k].split()]
                        #print expanded2
                        r1 = sum(model.wv[expanded1[i]] for i in range(len(expanded1)) )
                        r1max = max(model.wv[expanded1[i]] for i in range(len(expanded1)))
                        #print r1
                        r1 = r1/(len(expanded1))
                        r1 = r1/r1max
                        r2 = sum(model.wv[expanded2[i]] for i in range(len(expanded2)) )
                        r2max = max(model.wv[expanded2[i]] for i in range(len(expanded2)))
                        #print r2
                        r2 = r2/(len(expanded2))
                        r2 = r2/r2max
                        ans = ans + (r1-r2)
                    except:
                        ans = ans + 0
        ans = ans / (len(skills))
        #print ans
        if(ans > 0.5):
            arr.append(1)
        else:
            arr.append(0)
        ranking.append(ans)

#print(user_df.iloc[0]['UserID'])

name_array = []
for i in range(len(user_df)):
    for j in range(len(df)):
        name_array.append(user_df.iloc[i]['UserID'])

comp_array = []
r=0
for i in range(len(user_df)):
    for j in range(len(df)):
        comp_array.append(df.iloc[j]['Name'])
    


res = [
    ('Name of user',name_array ),
    ('Competition Name' , comp_array),
    ('Label',arr),
    ('Ranking',ranking),
    ]
result = pd.DataFrame.from_items(res)

#print(result)
result.to_csv("result22.csv")

