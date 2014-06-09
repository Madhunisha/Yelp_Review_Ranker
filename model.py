__author__ = 'Nikhil'
import pandas as pd
import numpy as np
import time
import datetime
import random
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from itertools import repeat
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
import sys
sys.stdout = open('output.log', 'w')

def rsmle(train,test):
    return np.sqrt(np.mean((pow(np.log(test+1) - np.log(train+1),2))))

def prepare_data(train_fl, guess_user_votes_fl=False):

    user = pd.read_csv('Data\\yelp_training_set_user.csv').set_index('user_id')
    user = user.drop(['type','name'],axis=1)
    user_test = pd.read_csv('Data\\yelp_test_set_user.csv').set_index('user_id').drop(['type','name'],axis=1)
    user = user.append(user_test)

    business = pd.read_csv('DataProcessed\\test_and_train_business_fea.csv').set_index('business_id')
    business= business.drop(['categories'],axis=1)

    if train_fl:
        review = pd.read_csv('DataProcessed\\review_fea.csv').set_index('review_id')
    else:
        review = pd.read_csv('DataProcessed\\review_test_fea.csv').set_index('review_id')
    ## JOIN DATA
    review = review.join(
        user, on='user_id', rsuffix='_user').join(
            business, on='business_id', rsuffix='_store').drop(
                ['user_id','business_id','text'],axis=1)

    review['w_count_store'] = review['review_count_store'] / np.float32(review['check_all']+1)
    review['w_count'] = review['review_count'] / np.float32(review['check_all']+1)
    ## same for user numbers of review vs numebrs of checkins
    review['w_count'] = review['w_count'].fillna(0)
    review['w_count_store'] = review['w_count_store'].fillna(0)

    ## difference between average user stars and current review
    ## difference between business stars and current review
    review['delta_user_review_star'] = abs((review['average_stars'] - review['stars']).fillna(0))
    review['delta_review_business_star'] = abs((review['stars_store'] - review['stars']).fillna(0))

    if train_fl:
        for col in ['votes_cool','votes_funny','votes_useful_user']:
            review['w_'+col] = review[col] / review['review_count']
        NA_user_fields = ['average_stars','review_count','votes_cool','votes_funny','votes_useful_user',
                          'w_votes_cool','w_votes_funny','w_votes_useful_user']
    else:
        for col in ['votes_cool','votes_funny','votes_useful']:
            review['w_'+col] = review[col] / review['review_count']
        NA_user_fields = ['average_stars','review_count','votes_cool','votes_funny','votes_useful',
                          'w_votes_cool','w_votes_funny','w_votes_useful']
    for col in NA_user_fields:
        review[col] = review[col].fillna(-1)


    return review

if __name__ == '__main__':
    print "PREPARING DATA..."
    review = prepare_data(True)

    ## Columns to drop during training phase Adjust drop columns to vary performance
    c_drop = ['cat_clust_60','cat_clust_70','cat_clust_35','cat_clust_20','cat_clust_50'
              ,'open','stem_unique_len_ratio','clust_200',
              'latitude', 'longitude',
              'loc_clust_5','loc_clust_10','loc_clust_15','loc_clust_20','loc_clust_25','loc_clust_30','loc_clust_40']
    review = review.drop(c_drop,axis=1)


    review_valid = review.ix[:int(len(review)*0.01),:]
    review = review.ix[set(review.index) - set(review_valid.index),:]
    print "Traing size: %i | Validation size: %i" % (len(review), len(review_valid))

    ## Setting up Regressors. Uncomment to test different regressors
    esti = 400; dep = 7
    gb = GradientBoostingRegressor(n_estimators=esti, max_depth=dep, random_state=7)
    # gb = linear_model.SGDRegressor(learning_rate='optimal',loss='huber',random_state=7)
    # gb = AdaBoostRegressor(n_estimators=500,loss='linear',random_state=7)
    # gb = linear_model.Lasso(precompute=True,positive=True,max_iter=500)
    init_time = time.time()
    print "Training..",
    gb.fit(review.drop('votes_useful',axis=1), np.log(review['votes_useful']+1))
    print "DONE! %.2fm" % ((time.time()-init_time)/60)

    print "Train: "+str(rsmle(np.exp(gb.predict(review.drop(['votes_useful'],axis=1))) - 1, review['votes_useful']))
    print "CV: "+str(rsmle(np.exp(gb.predict(review_valid.drop(['votes_useful'],axis=1))) - 1, review_valid['votes_useful']))

    print " Start Prediction"
    test = prepare_data(False).drop(c_drop, axis=1)
    preds = np.exp(gb.predict(test))-1
    preds[preds<0] = 0
    pd.Series(preds,index=test.index,name='Votes').to_csv('submission.csv', header=True)


    feature_importance = gb.feature_importances_

    feature_list=list(review.columns.values)
    print "feature list------------------"
    print feature_list
    print "feature importance---------------------"
    print feature_importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    print "New feature importance"
    print feature_importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1, 2, 2)
    pl.barh(pos, feature_importance, align='center')
    pl.yticks(pos, feature_list)
    pl.xlabel('Relative Importance')
    pl.title('Variable Importance')
    pl.show()

