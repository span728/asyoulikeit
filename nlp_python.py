import pandas as pd 

data = pd.read_csv('ayli_text.csv', error_bad_lines=False);
data_text = data[['act_text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))

import gensim 
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np 
np.random.seed(2018)

# some custom stopwords
customStopwords = ["shall", "thee", "thou", "le", "thy", "well", "go", "yet", "come", "enter", "hath", "first", "good", "man", "one", "say", "take", "see", "make", "might", "tis", "sir", "let", "us", "upon", "every", "may", "must", "de", "quoth", "sans"]


import nltk
nltk.download('wordnet')

# lemmatize example
# print(WordNetLemmatizer().lemmatize('seen', pos='v'))
# end lemmatize example

# stemmer example
stemmer = SnowballStemmer('english')
# original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
#            'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
#            'traditional', 'reference', 'colonizer','plotted']
# singles = [stemmer.stem(plural) for plural in original_words]
# print(pd.DataFrame(data = {'original word': original_words, 'stemmed': singles}))

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in customStopwords and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
#end stemmer example

# time to process the docs 
processed_docs = documents['act_text'].map(preprocess)

print(processed_docs[:10])

# bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# print("ehello")
# bow_doc_4310 = bow_corpus[1371]
# for i in range(len(bow_doc_4310)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
#                                                      dictionary[bow_doc_4310[i][0]], 
#                                                      bow_doc_4310[i][1]))

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


