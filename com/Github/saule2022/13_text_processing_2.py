import pandas as pd
import nltk
import pathlib
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix

en_stopwords = stopwords.words('english')
data_path = pathlib.Path(r'C:\DATA\various_datasets\fake_news\train.csv')
test_path = pathlib.Path(r'C:\DATA\various_datasets\fake_news\test.csv')
labels_path = pathlib.Path(r'C:\DATA\various_datasets\fake_news\submit.csv')

df = pd.read_csv(data_path.as_posix())


def normalize(input_string):
    return re.sub('[^a-z ]', '', input_string.lower())


def remove_stopwords(input_string, stopwords=en_stopwords):
    return ' '.join([word for word in input_string.split() if word not in stopwords])


# def remove_stopwords_v2(input_string, stopwords=en_stopwords):
#     words = input_string.split()
#     out = []
#     for word in words:
#         if word not in stopwords:
#             out.append(word)
#     output = ' '.join(out)
#     return output

df['text'] = df['text'].astype('str')
df['normalize'] = df['text'].apply(lambda x: normalize(x))
df['clean_text'] = df['normalize'].apply(lambda x: remove_stopwords(x))

vectorizer = CountVectorizer(analyzer='word')
train_vect = vectorizer.fit_transform(df['clean_text'])

svd = TruncatedSVD(n_components=10000)
svd_features = svd.fit_transform(train_vect)

# model = NMF(n_components=5)
# model.fit(train_vect)
#
# # print(model.components_)
#
# feature_names = vectorizer.get_feature_names()
# for topic_idx, topic in enumerate(model.components_):
#         top_features_ind = topic.argsort()[:-11:-1]
#         top_features = [feature_names[i] for i in top_features_ind]
#         weights = topic[top_features_ind]
#         print('\n\n')
#         print(top_features)
#         print(weights)

RF = RandomForestClassifier(n_jobs=-1, n_estimators=300)
RF.fit(svd_features, df['label'])

test = pd.read_csv(test_path.as_posix())
labels = pd.read_csv(labels_path.as_posix())
test['text'] = test['text'].astype('str')
test['normalize'] = test['text'].apply(lambda x: normalize(x))
test['clean_text'] = test['normalize'].apply(lambda x: remove_stopwords(x))

test_vect = vectorizer.transform(test['clean_text'])
svd_test_features = svd.transform(test_vect)

predictions = RF.predict(svd_test_features)

print(classification_report(labels['label'], predictions))
print(confusion_matrix(labels['label'], predictions))