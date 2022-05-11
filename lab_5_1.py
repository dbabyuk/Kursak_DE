import pandas as pd
import numpy as np
import textblob
from sklearn.feature_extraction.text import CountVectorizer
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

# Setting randomizer for data sampling
rd = np.random.RandomState(seed=100)


# lemmatizer = TextBlob()

# Read the input data
data_raw = pd.read_csv('twitter-airline-sentiment.csv', encoding_errors='ignore')
# Leaving only data if gender is defined
data_raw = data_raw[data_raw['airline_sentiment'] != 'neutral']

n_total = 1000
rows = data_raw.shape[0]
rd_ind_list = rd.randint(0, rows-1, n_total)
data_init = data_raw[['airline_sentiment', 'text']].iloc[rd_ind_list]


def clean_txt(bulk: list):
    patterns_to_remove = [r"\bhttps:\//[a-z0-9.]*/[a-z0-9]*", r"\bhttp:\//[a-z0-9.]*/[a-z0-9]*", "@\\w+", "\\d+"]
    res = []
    for sentence in bulk:
        for pattern in patterns_to_remove:
            sentence = re.sub(pattern, '', sentence)
        res.append(sentence)
    return res


def preprocess(bulk: list):
    res = []
    for sentence in bulk:
        row_blob = textblob.TextBlob(sentence).words.lemmatize().stem()
        res.append(' '.join(row_blob))
    return res

corpus = preprocess(clean_txt(data_init.text))

text_train, text_value, y_train, y_value = \
    train_test_split(corpus, data_init.airline_sentiment, test_size=0.3, random_state=100)


vect = CountVectorizer(stop_words='english', min_df=3, ngram_range=(1, 1))


X_train_df3 = vect.fit_transform(text_train)
X_val_df3 = vect.transform(text_value)


lr = LogisticRegressionCV(max_iter=300)
lr.fit(X_train_df3, y_train)

print('C_', lr.C_)
print('Score = ', lr.score(X_val_df3, y_value))

print()
X = vect.get_feature_names_out()
print('X = ', X)

vocab = vect.vocabulary_