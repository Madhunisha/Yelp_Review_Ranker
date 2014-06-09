from __future__ import division
__author__ = 'Nikhil'

from dateutil.parser import parse
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer


use_stop_words = True
min_word_len = 3
snapshot_date = datetime.datetime(2013,1,20)


print "loading review data from csv file..."
review = pd.read_csv('Data\\yelp_training_set_review.csv',
                     converters={'date': parse}).set_index('review_id')
review = review.drop(['type','votes_cool','votes_funny'],axis=1)
review['text'] = review['text'].fillna("")
review['review_len'] = review['text'].apply(len)


tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()
print "Extracting Stopwords from NLTK corpus"
stopset = set(stopwords.words('english'))




print "stemming words "
stemmedReview = []
init_time = time.time()
for i in range(len(review)):
    stemmedReview.append(
        [stemmer.stem(word) for word in [w for w in tokenizer.tokenize(review.ix[i,'text'].lower()) if (len(w) > min_word_len and w not in stopset)]]
    )


print (time.time()-init_time)/60

review['text'] = stemmedReview
del stemmedReview
review['stem_len'] = review['text'].apply(len)
review['stem_unique_len'] = review['text'].apply(np.unique).apply(len)
review['text'] = [' '.join(words) for words in review['text']]


vect = HashingVectorizer(stop_words='english',non_negative=True,ngram_range=(2,2),decode_error='ignore',n_features=20000)

stem_fea = vect.fit_transform(review['text'])
print "extracting feature"
init_time = time.time()
print (time.time()-init_time)/60
print ("Done")
print("must be dump error")

joblib.dump(vect,'Models\\review_vect') ## DUMP vect AND UNALLOCATE
del vect

from sklearn.cluster import MiniBatchKMeans
for esti in (200,300,500,750):
    km = MiniBatchKMeans(n_clusters=esti, random_state=500,init_size=esti*3)

    print "fitting "+str(esti)+" clusters"
    init_time = time.time()
    km.fit(stem_fea)
    print (time.time()-init_time)/60

    review['clust_'+str(esti)] = km.predict(stem_fea)
    joblib.dump(km,'Models\\review_km'+str(esti))


def get_days(row):
    return row.days

review['stem_unique_len_ratio'] = review['stem_unique_len'] / review['stem_len']
review['date'] = snapshot_date - review['date']
review['date'] = review['date']/np.timedelta64(1,'D')

review.to_csv('DataProcessed\\review_fea.csv')
