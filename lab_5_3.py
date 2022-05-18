import pandas as pd
import numpy as np
import textblob
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
from sklearn.preprocessing import Normalizer
import re
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "debates\\clinton_1st.txt"
# Read the bulk data
with open(f'election_2016\\{file_name}', 'r') as file:
    data_raw = file.read()
    file.close()

data_blob = textblob.TextBlob(data_raw)


# Sentiment Analysis
def distribute_sentiment(data_):
    """Returns DataFrame with sentiment polarity and score for each sentence"""
    sentences = data_.sentences
    scores = [sentence.polarity for sentence in sentences]
    sentiments = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in scores]
    return pd.DataFrame({'sentence': sentences,
                         'score': scores,
                         'sentiment': sentiments})


sentiments_df = distribute_sentiment(data_blob)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(['neg', 'neu', 'pos'], [1, 2, 4], color=['r', 'b', 'g'])
plt.show()


def clean_txt(bulk: str):
    """Removes words from each sentence defined by RegEx"""
    patterns_to_remove = ["\\d+"]
    for pattern in patterns_to_remove:
        bulk = re.sub(pattern, '', bulk)
    return bulk


data_cleaned = clean_txt(data_raw)


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


MY_STOP_WORDS = ["'s", "'ve", "'re", "'m", "'ll", "'d"]


def wc_df(bulk: str):
    """Processing words through lemmatization and stemming"""
    blob = textblob.TextBlob(bulk).lower()
    word_list = []
    for word in blob.words:
        if word not in my_stop_words(MY_STOP_WORDS):
            word_list.append(word.lemmatize('v'))
    new_blob = textblob.TextBlob(' '.join(word_list))
    wc = new_blob.word_counts
    tags = dict(new_blob.tags)
    df = pd.DataFrame({'word': wc.keys(), 'count': wc.values(), 'tags': None})
    # Adding tags to the data frame
    for word in df['word']:
        df.loc[df['word'] == word, ['tags']] = tags[word]
    return df.sort_values(['count'], ascending=False)


word_count_df = wc_df(data_cleaned)


nouns_ordered = word_count_df[word_count_df['tags'].isin(('NN', 'NNS'))]


def stem_plotter(input_data, amount=10):
    """Plots stem for particular class and stem number"""
    data_ = input_data.iloc[:amount]
    length = amount
    plt.stem(data_['count'], markerfmt='ro')
    plt.xlabel('Words')
    plt.xticks(rotation=45, ticks=range(length), labels=data_['word'])
    plt.title(f"File: {file_name}")
    plt.show()

stem_plotter(nouns_ordered)


def np_df(bulk: str):
    """Processing words through lemmatization and stemming"""
    blob = textblob.TextBlob(bulk).lower()
    wc = blob.np_counts
    df = pd.DataFrame({'word': wc.keys(), 'count': wc.values()})
    return df.sort_values(['count'], ascending=False)

phrases = np_df(data_raw)