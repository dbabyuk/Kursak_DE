import pandas as pd
import numpy as np
import textblob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import _stop_words
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read the bulk data
data_raw = pd.read_csv('twitter-airline-sentiment.csv', encoding_errors='ignore')
# Dropping data for airline_sentiment='neutral', thus leaving only two classes: positive, negative
data_raw = data_raw[data_raw['airline_sentiment'] != 'neutral']

# Defining data sample size
n_total = 5000
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

# Setting vectorization procedure
vect = CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS), min_df=2,
                       ngram_range=(1, 1))

# Words vectorization
X_train = vect.fit_transform(text_train)
X_val = vect.transform(text_value)

# Model definition for (Logistic Regression, Bayes Naive etc) data training
train_approach = LogisticRegressionCV(max_iter=500)
# Training the model
train_approach.fit(X_train, y_train)

# All words included into vectorization
words = vect.get_feature_names_out()
# Frequency of vectorized words
word_freq = X_train.sum(axis=0).tolist()[0]
_words_vect = vect.transform(words)

# Determining class (positive/negative) for each word among words
words_class = train_approach.predict(_words_vect).tolist()

# Formation summary data frame for the processed words
summary_table = pd.DataFrame({'token': words, 'freq': word_freq, 'class': words_class})


def bag_of_words(class_, number=15, freq='max', data_=summary_table):
    """Returns DataFrame for specific bag with words sorted by frequency"""
    if freq == 'max':
        ascending = False
    elif freq == 'min':
        ascending = True
    else:
        ascending = None
    class_slice = data_[data_['class'] == class_].sort_values(['freq'], ascending=ascending).iloc[:number]
    return class_slice.sort_values(['freq'], ascending=ascending)


def stem_plotter(data_, class_):
    """Plots stem for particular class and stem number"""
    length = data_.shape[0]
    plt.stem(data_['freq'], markerfmt='ro' if class_ == 'negative' else 'go')
    plt.xlabel('Words')
    plt.xticks(rotation=90, ticks=range(length), labels=data_['token'])
    plt.title(f"Class: {class_.upper()}")
    plt.show()

# Word bags displaying
bag_positives = bag_of_words('positive')
bag_negatives = bag_of_words('negative')
print('Positive Bag of words:')
print(bag_positives)
print()
print('Negative Bag of words:')
print(bag_negatives)
print()

# Visualization the word bags
stem_plotter(bag_positives, 'positive')
stem_plotter(bag_negatives, 'negative')


# Prediction of the output for test values
y_pred = train_approach.predict(X_val)

# Metrics Scores calculation
acc = accuracy_score(y_value, y_pred)
precision = precision_score(y_value, y_pred, average='weighted')
recall = recall_score(y_value, y_pred, average='weighted')
f1 = f1_score(y_value, y_pred, average='weighted')
print()
print('Accuracy score = ', acc)
print('Precision score = ', precision)
print('Recall score = ', recall)
print('F1 score = ', f1)
print()


# TFiD Words vectorization
tfid_vect = TfidfVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS), min_df=2,
                       ngram_range=(1, 1))
X_train_tfid = tfid_vect.fit_transform(text_train)
X_val_tfid = tfid_vect.transform(text_value)

train_approach.fit(X_train_tfid, y_train)

# Metrics Scores calculation for Tfid
y_pred_tfid = train_approach.predict(X_val_tfid)
acc_tfid = accuracy_score(y_value, y_pred_tfid)
precision_tfid = precision_score(y_value, y_pred_tfid, average='weighted')
recall_tfid = recall_score(y_value, y_pred_tfid, average='weighted')
f1_tfid = f1_score(y_value, y_pred_tfid, average='weighted')
print()
print('Tfid Accuracy score = ', acc_tfid)
print('Tfid Precision score = ', precision_tfid)
print('Tfid Recall score = ', recall_tfid)
print('Tfid F1 score = ', f1_tfid)

# Bigram computation
vect_bigram = CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS), min_df=2,
                       ngram_range=(2, 2))
X_train_bigram = vect_bigram.fit_transform(text_train)
X_val_bigram = vect_bigram.transform(text_value)
train_approach.fit(X_train_bigram, y_train)

acc_bigram = train_approach.score(X_val_bigram, y_value)
print()
print('Bigram accuracy score = ', acc_bigram)

# Triigram computation
vect_trigram = CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS), min_df=2,
                       ngram_range=(3, 3))
X_train_trigram = vect_trigram.fit_transform(text_train)
X_val_trigram = vect_trigram.transform(text_value)
train_approach.fit(X_train_trigram, y_train)

acc_trigram = train_approach.score(X_val_trigram, y_value)
print()
print('Trigram accuracy score = ', acc_trigram)
