import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords

#stop = stopwords.words('english')
#stop.pop(stop.index('and'))

train = ['This is the first document.',
         'This document is the second document.',
         'And this is the third one.',
         'Is this the first document?']

countvectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
tfidfvectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,2))

count_wm = countvectorizer.fit_transform(train)
tfidf_wm = tfidfvectorizer.fit_transform(train)

count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()

df_countvect = pd.DataFrame(data=count_wm.toarray(), index=['Doc1','Doc2','Doc3','Doc4'], columns=count_tokens)
df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), index=['Doc1','Doc2','Doc3','Doc4'], columns=tfidf_tokens)

print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)

