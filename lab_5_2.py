import pandas as pd
import numpy as np
import textblob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.preprocessing import Normalizer
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read the bulk data
data_raw = pd.read_csv('twitter-airline-sentiment.csv', encoding_errors='ignore')
# Dropping data for airline_sentiment='neutral', thus leaving only two classes: positive, negative
data_raw = data_raw[data_raw['airline_sentiment'] != 'neutral']

# Defining data sample size
n_total = 1000
rows = data_raw.shape[0]
# Setting randomizer with specific seed value
rd = np.random.RandomState(seed=100)
rd_ind_list = rd.randint(0, rows - 1, n_total)
# Forming working data set (data_init) with two columns: 'airline_sentiment' and 'text'
data_init = data_raw[['airline_sentiment', 'text']].iloc[rd_ind_list]


def clean_txt(bulk: list):
    """Removes words from each sentence defined by RegEx"""
    patterns_to_remove = [r"\bhttps:\//[a-z0-9.]*/[a-z0-9]*", r"\bhttp:\//[a-z0-9.]*/[a-z0-9]*", "@\\w+", "\\d+"]
    res = []
    for sentence in bulk:
        for pattern in patterns_to_remove:
            sentence = re.sub(pattern, '', sentence)
        res.append(sentence)
    return res


def preprocess(bulk: list):
    """Processing words through lemmatization and stemming"""
    res = []
    for sentence in bulk:
        row_blob = textblob.TextBlob(sentence).words.lemmatize().stem()
        res.append(' '.join(row_blob))
    return res


# Forming corpus after data preprocessing
corpus = preprocess(clean_txt(data_init.text))

# Splitting initial data into training and test groups
text_train, text_value, y_train, y_value = train_test_split(corpus, data_init.airline_sentiment, test_size=0.3,
                                                            random_state=100)

def my_stop_words(words_to_add=None, words_to_remove=None):
    """Returns set of STOP_WORDS"""
    my_stop_words = set(_stop_words.ENGLISH_STOP_WORDS)
    if words_to_add:
        for word in words_to_add:
            my_stop_words.add(word)
    if words_to_remove:
        for word in words_to_remove:
            my_stop_words.remove(word)
    return my_stop_words


MY_STOP_WORDS = ['flight', 'aa', 'wa', 'thi', 've', 'just', 'air', 'plane', 'tri']


# Grid setup for optimal parameters search
param_grid = {'logisticregression__C': [10, 1, 0.1, 0.001],
              'countvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
              'countvectorizer__min_df': [1, 2, 3],
              'normalizer': [None, Normalizer()]
              }

grid = GridSearchCV(make_pipeline(CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS)),
                                  Normalizer(), LogisticRegression(), memory='cache_folder'),
                    param_grid=param_grid, cv=10)
grid.fit(text_train, y_train)

# Grid results output
best_score = grid.best_score_
print('Best Score = ', best_score)
best_params = grid.best_params_
print('Grid best params:')
print(best_params)

# Use of optimal grid parameters in prediction
vect = CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS),
                       min_df=best_params['countvectorizer__min_df'],
                       ngram_range=best_params['countvectorizer__ngram_range'])
X_train = vect.fit_transform(text_train)
X_val = vect.transform(text_value)

lr = LogisticRegression(C=best_params['logisticregression__C'])
lr.fit(X_train, y_train)

# Prediction of the output for test values
y_pred = lr.predict(X_val)

# Metrics Scores calculation
acc = accuracy_score(y_value, y_pred)
precision = precision_score(y_value, y_pred, average='weighted')
recall = recall_score(y_value, y_pred, average='weighted')
f1 = f1_score(y_value, y_pred, average='weighted')
print()
print('Results for optimal grid parameters')
print('Accuracy score = ', acc)
print('Precision score = ', precision)
print('Recall score = ', recall)
print('F1 score = ', f1)
print()
