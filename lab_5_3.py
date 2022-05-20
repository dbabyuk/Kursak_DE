"""It consists of 2 parts. First part deals with sentiment analysis for Clinton/Trump 2016 election debate.
The user is prompted to select one of three debate numbers. Second part performs word analysis and topic modelling"""
import pandas as pd
import numpy as np
import textblob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words
import re
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# --- PART 1. Sentiment Analysis---------------------

CANDIDATES = ['clinton', 'trump']
# Interactive input debate number
# debate_number = int(input('Enter number (1, 2, 3) of Clinton/Trump election2016 debate '))
debate_number = 2
end_str = '1st' if debate_number == 1 else '2nd' if debate_number == 2 else '3rd'


def candidate_data(name):
    """Reads file content for candidate debate speech"""
    file_name = f"debates\\{name}_{end_str}.txt"
    with open(f'election_2016\\{file_name}', 'r') as file:
        data_read = file.read()
        file.close()
    return data_read


# Text data for each candidate read from file
data_raw = {name: candidate_data(name) for name in CANDIDATES}


def sentiment_analysis(names: list):
    """Splits text content into textblob sentences; collects their polarity and subjectivity score;
     defines its sentiment into classes. Returns results in DataFrame format"""
    candidates = []
    sentences = []
    polarity_scores = []
    polarities = []
    subjectivities = []
    subjectivity_scores = []
    for candidate in names:
        data_str = data_raw[candidate]
        _sentences = textblob.TextBlob(data_str).sentences
        _polarity_scores = [sentence.polarity for sentence in _sentences]
        _subjectivity_scores = [sentence.subjectivity for sentence in _sentences]
        sentences += _sentences
        candidates += [candidate] * len(_sentences)
        polarity_scores += _polarity_scores
        polarities += ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in
                       _polarity_scores]
        subjectivities += ['high' if score >= 0.5 else 'low' for score in _subjectivity_scores]
        subjectivity_scores += _subjectivity_scores
    return pd.DataFrame(
        {'candidate': candidates, 'sentence': sentences, 'polarity_score': polarity_scores, 'polarity': polarities,
         'subjectivity_score': subjectivity_scores, 'subjectivity': subjectivities
         })


sentiment_results = sentiment_analysis(CANDIDATES)

# Output of positive class for both candidates
for name in CANDIDATES:
    print()
    print(f'The most positive phrase for {name.capitalize()}:')
    row_selected = sentiment_results[sentiment_results['candidate'] == name].sort_values(by='polarity_score', ascending=False)
    print(row_selected['sentence'].iloc[0])
    print('ITS POLARITY SCORE = ', row_selected['polarity_score'].iloc[0])

# Output of negative class for both candidates
for name in CANDIDATES:
    print()
    print(f'The most negative phrase for {name.capitalize()}:')
    row_selected = sentiment_results[sentiment_results['candidate'] == name].sort_values(by='polarity_score')
    print(row_selected['sentence'].iloc[0])
    print('ITS POLARITY SCORE = ', row_selected['polarity_score'].iloc[0])


def sentiment_freq(sentiments_df, sentiment_type):
    """Based on sentiment_analysis counts number of classes for each candidate name.
    Returns DataFrame with normalized class frequencies"""
    cand_values = dict(sentiments_df['candidate'].value_counts())
    frames = []
    for cand in cand_values.keys():
        sentiments_ = sentiments_df[sentiments_df['candidate'] == cand][sentiment_type].value_counts(normalize=True)
        frames.append(pd.DataFrame({cand: sentiments_}).T)
    merged_df = pd.concat(frames)
    return merged_df

#
polaritiy_results = sentiment_freq(sentiment_results, 'polarity')
subjectivity_results = sentiment_freq(sentiment_results, 'subjectivity')


def bar_plotter(names: list, type_: str, data_):
    """Plots normalized sentiment frequencies bars for each candidate name"""
    freq = data_
    fig = plt.figure()
    axes = fig.add_subplot(111)
    x_cluster = {names[0]: 2, names[1]: 7}
    _sentiment_info = {'polarity': {'negative': {'shift': 0, 'color': 'r'},
                                    'neutral': {'shift': 1, 'color': 'b'},
                                    'positive': {'shift': 2, 'color': 'g'}},
                       'subjectivity': {'low': {'shift': 0, 'color': 'r'},
                                        'high': {'shift': 1, 'color': 'g'}}
                       }
    sentiment_type = _sentiment_info[type_]
    ticks = []
    for candidate in names:
        for bar in sentiment_type.keys():
            shift = sentiment_type[bar]['shift']
            color_ = sentiment_type[bar]['color']
            axes.bar(x_cluster[candidate] + shift, freq.loc[candidate][bar], color=color_)
        bar_x_ticks = range(x_cluster[candidate], len(sentiment_type.keys()) + x_cluster[candidate])
        axes.set_xticks(bar_x_ticks, sentiment_type, rotation=90)
        ticks += list(bar_x_ticks)
        axes.text(x_cluster[candidate], 0.8, candidate.capitalize(), fontsize="xx-large")
    axes.set_xticks(ticks, list(sentiment_type.keys()) * 2, rotation=30)
    axes.set_ymargin(0.8)
    plt.ylabel(f'Normalized {type_}', rotation=90)
    plt.title(f'{end_str} {names[0].capitalize()}/{names[1].capitalize()} Debate')
    plt.show()


# Results visualization for class frequencies
bar_plotter(CANDIDATES, 'polarity', polaritiy_results)
bar_plotter(CANDIDATES, 'subjectivity', subjectivity_results)


# ---- PART 2. Nouns and topics modelling

def clean_txt(bulk: dict):
    """Removes some junk words from each sentence defined by RegEx patterns"""
    patterns_to_remove = ["\\d+"]
    for key, value in bulk.items():
        for pattern in patterns_to_remove:
            bulk[key] = re.sub(pattern, '', value)
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


MY_STOP_WORDS = ["'s", "'ve", "'re", "'m", "'ll", "'d", "ve", "ll", "ha", "wa", "maybe", "just",
                 "donald", "ca", "got", "sure", "did"]


def wc_df(bulk: str):
    """Processing words through lemmatization and tagging. Returns DataFrame with sorted word counts"""
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


def noun_frequecies(data_: dict):
    """Returns 10 the most frequent nouns from the DataFrame"""
    res = {}
    for candidate in CANDIDATES:
        word_count_df = wc_df(data_[candidate])
        nouns_ordered = word_count_df[word_count_df['tags'].isin(('NN', 'NNS'))]
        res[candidate] = nouns_ordered[['word', 'count']].iloc[:10]
    return res


nouns = noun_frequecies(data_cleaned)


# Noun output
for candidate in CANDIDATES:
    print()
    print(f"The most frequent nouns for {candidate.capitalize()} speech:")
    print(nouns[candidate]['word'].tolist())


def preprocess(bulk: str):
    """Processing words via lemmatization"""
    res = []
    sentences = textblob.TextBlob(bulk).sentences
    for sentence in sentences:
        row_blob = sentence.words.lemmatize()
        res.append(' '.join(row_blob))
    return res


def get_topics(data_: dict, number_of_topics=3, words_per_topic=10):
    """SVD analysis for topic modelling. Returns dict of number_of_topics vectors"""
    res = {}
    for candidate in CANDIDATES:
        text_data = preprocess(data_[candidate])
        vect = CountVectorizer(stop_words=my_stop_words(words_to_add=MY_STOP_WORDS), min_df=4)
        vector_data = vect.fit_transform(text_data).todense()
        svd_modeling = TruncatedSVD(n_components=number_of_topics)
        svd_modeling.fit(vector_data)
        components = svd_modeling.components_
        vocab = vect.get_feature_names_out()
        topic_word_list = []
        for ind in range(components.shape[0]):
            terms_comp = zip(vocab, components[ind])
            sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:words_per_topic]
            topic = dict(sorted_terms)
            topic_word_list.append(topic)
        res[candidate] = topic_word_list
    return res


topics = get_topics(data_cleaned)


def topics_output(data_):
    """Prints and plots (stem) results for topics"""
    for cand in CANDIDATES:
        ind = 1
        cand_topics = data_[cand]
        print()
        print(f'Topics of {cand.capitalize()} speech:')
        for topic in cand_topics:
            print(f'Topic {ind}:', ' '.join(topic.keys()))
            plt.stem(topic.values())
            plt.xticks(rotation=45, ticks=range(len(topic)), labels=topic.keys())
            plt.title(f'Topic {ind} of {cand.capitalize()} speech')
            plt.show()
            ind += 1

# Visualization of topic modelling
topics_output(topics)
