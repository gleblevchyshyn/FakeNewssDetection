import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('preparedData.csv')

vec = TfidfVectorizer()

s = vec.fit_transform(df['title'].values.astype('U')).toarray()

clf = MultinomialNB()
clf.fit(s, df['label'])

