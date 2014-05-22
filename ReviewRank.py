from _ctypes import sizeof

__author__ = 'Nikhil'

import json
import numpy
from sklearn import svm


# data_user = {}
# data_review = []
# with open('yelp_training_set_user.json') as f:
#     for line in f:
#         ind = json.loads(line)
#         data_user[ind['user_id']] = ind
# cnt = 0
# with open('yelp_training_set_review.json') as f:
#     for line in f:
#         ind = json.loads(line)
#         data_review.append(ind)
#         cnt += 1
#         if cnt > 1000:
#             break


class YelpReview:
    def __init__(self):
        self.data_review = []
        self.data_user = {}
        self.data_business = {}
        self.data_checkin = {}

    def initialize(self,reviewFile,userFile,businessFile,checkinFile):
        cnt = 0
        with open(reviewFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_review.append(ind)
                cnt += 1
                if cnt > 2000:
                    break
        with open(userFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_user[ind['user_id']] = ind
        with open(businessFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_business[ind['business_id']] = ind
        with open(businessFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_business[ind['business_id']] = ind

    def BusinessFeatures(self,id):
        business = self.data_business[id]
        return business['stars'],business['review_count']


    def UserFeatures(self,id):

        userFeat = [0,0,0,0,0,0]
        if(self.data_user.has_key(id)== False):
            return userFeat
        user = self.data_user[id]
        userFeat=[]
        #print id
        userFeat.append(user['review_count'])
        userFeat.append(user['average_stars'])
        userFeat.append(user['votes']['useful'])
        userFeat.append(user['votes']['funny'])
        userFeat.append(user['votes']['cool'])
        userFeat.append(float(userFeat[2]/userFeat[0]))
        return userFeat

    def PopulateMatrix(self):
        matrix = numpy.zeros((len(self.data_review),9))
        target = numpy.zeros((len(self.data_review)))
        cnt = 0
        for review in self.data_review:
            matrix[cnt][:2] = self.BusinessFeatures(review['business_id'])
            matrix[cnt][2:8] = self.UserFeatures(review['user_id'])
            target[cnt] = review['votes']['useful']
            cnt = cnt +1
        return matrix,target

    

def main():
    totalData = YelpReview()
    totalData.initialize('yelp_training_set_review.json','yelp_training_set_user.json','yelp_training_set_business.json','yelp_training_set_checkin.json')
    print totalData.data_review[20]
    matrix , tar = totalData.PopulateMatrix()
    print "population done"
    reg = svm.SVR(kernel = 'rbf')
    reg.fit(matrix[:1700],tar[:1700])
    print "classifier modeled"
    for x in xrange(1600,1900):
        predictions = reg.predict([matrix[x]])
        print "value :" + str(totalData.data_review[x]['votes']['useful']) + '|' + 'prediction :' + str(round(predictions))
        print "------------------------------------"


main()