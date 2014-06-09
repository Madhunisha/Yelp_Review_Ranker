from _ctypes import sizeof

__author__ = 'Nikhil'

import json
import numpy
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn import linear_model

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
                # if cnt > 100000:
                #     break
        with open(userFile) as f:
            for line in f:
                ind = json.loads(line)
                ind["status"] = "old"
                self.data_user[ind['user_id']] = ind
        with open(businessFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_business[ind['business_id']] = ind
        with open(checkinFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_checkin[ind['business_id']] = ind

    def BusinessFeatures(self,id):
        business = self.data_business[id]
        return business['stars'],business['review_count']


    def UserFeatures(self,id):
        userFeat = [0,0,0,0,0,0]
        if(self.data_user.has_key(id)== False):
            print "mismatch"
            return userFeat
        user = self.data_user[id]
        userFeat=[]
        #print id
        userFeat.append(user['review_count'])
        userFeat.append(user['average_stars'])
        userFeat.append(user['votes']['useful'])
        userFeat.append(user['votes']['funny'])
        userFeat.append(user['votes']['cool'])
        userFeat.append(userFeat[2]/float(userFeat[0]))
        return userFeat

    def ReviewFeatures(self,review):
        reviewFeat = [0,0,0]



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



    def Preprocess(self):
        user = {"status": "new", "votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": " ", "name": "Unknown", "average_stars": 0.0, "review_count": 1, "type": "user"}
        count = 0
        for review in self.data_review:
            userID = review['user_id']
            if(self.data_user.has_key(userID)):
                user = self.data_user[userID]
                if(user['status'] == 'new'):
                    user['votes']['funny'] = user['votes']['funny'] + review['votes']['funny']
                    user['votes']['useful'] = user['votes']['useful'] + review['votes']['useful']
                    user['votes']['cool'] = user['votes']['cool'] + review['votes']['cool']
                    user['average_stars'] = (user['average_stars'] * user['review_count'] + review['stars'])/(user['review_count'] + 1)
                    user['review_count'] = user['review_count'] + 1
                    self.data_user[userID] = user
                    # count = count + 1
                    # print userID
            else:
                count = count + 1
                user['user_id'] = userID
                user['name'] = "Unknown"
                user['status'] = 'new'
                user['votes']['funny'] = review['votes']['funny']
                user['votes']['useful'] = review['votes']['useful']
                user['votes']['cool'] = review['votes']['cool']
                user['average_stars'] = review['stars']
                user['review_count'] = 1
                self.data_user[userID] = user
        return count

def maine():
    totalData = YelpReview()
    add = 'C:\Users\Nikhil\Documents\social_train'
    totalData.initialize(add+'\yelp_training_set_review.json',add+'\yelp_training_set_user.json',add+'\yelp_training_set_business.json',add+'\yelp_training_set_checkin.json')
    # count = totalData.DataOverlap('yelp_test_set_review.json')
    # print "count :" + str(count)
    count = totalData.Preprocess()
    print "count :" + str(count)
    matrix , tar = totalData.PopulateMatrix()
    X_train = normalize(matrix,norm='l1',axis=0)

    # X_train = min_max_scaler.fit_transform(matrix)
    print "population done"
    print "matrix size" + str(len(tar))
    # reg = SVR(kernel = 'poly', verbose= True)

    X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(matrix[:200000], tar[:200000])

    print "New feature" + str(X_new.shape) + "----" + str(X_new[0][:])
    print "old feature" + str(matrix.shape) + "----" + str(matrix[0][:])
    i = input("find features")

    reg = linear_model.SGDRegressor(loss='huber',learning_rate='constant')
    print "start fit"
    reg.fit(X_train[:220000],tar[:220000])
    print "classifier modeled"
    # predictions = []
    # for x in xrange(229100,229200):
    #      predictions.append(round(reg.predict([matrix[x]])))
    #      print "value :" + str(tar[x]) + '|' + 'prediction :' + str(predictions[x-229100])
    #      print "------------------------------------"
    # count =0
    # for i in xrange(100):
    #     if((predictions[i] >= tar[9100+i] and predictions[i] <= (tar[9100+i]) + 0.5 * (tar[9100+i])) or (predictions[i] <= tar[9100+i] and predictions[i] >= 0.5 * (tar[9100+i]))):
    #         count = count + 1
    #
    # print "percentage acc :" + str(float(count/100.00))

    predictions = []
    for x in xrange(8900,9000):
         predictions.append((reg.predict([matrix[x]])))
         print "value :" + str(tar[x]) + '|' + 'prediction :' + str(predictions[x-8900])
         print "------------------------------------"
    count =0
    for i in xrange(100):
        if((predictions[i] >= tar[8900+i] and predictions[i] <= (tar[8900+i]) + 0.5 * (tar[8900+i])) or (predictions[i] <= tar[8900+i] and predictions[i] >= 0.5 * (tar[8900+i]))):
            count = count + 1

    print "percentage acc train:" + str(float(count/100.00))

    predictions = []
    for x in xrange(9100,9200):
         predictions.append(round(reg.predict([matrix[x]])))
         print "value :" + str(tar[x]) + '|' + 'prediction :' + str(predictions[x-9100])
         print "------------------------------------"
    count =0
    for i in xrange(100):
        if(predictions[i] == tar[9100+i]):
            count = count + 1

    print "percentage acc :" + str(float(count/100.00))

    predictions = []
    for x in xrange(229000,229100):
         predictions.append(round(reg.predict([matrix[x]])))
         print "value :" + str(tar[x]) + '|' + 'prediction :' + str(predictions[x-229000])
         print "------------------------------------"
    count =0
    for i in xrange(100):
        if(predictions[i] == tar[229000+i]):
            count = count + 1
    print "percentage acc train:" + str(float(count/100.00))



def main():
    totalData = YelpReview()
    add = 'C:\Users\Nikhil\Documents\social_train'
    totalData.initialize(add+'\yelp_training_set_review.json',add+'\yelp_training_set_user.json',add+'\yelp_training_set_business.json',add+'\yelp_training_set_checkin.json')
    # count = totalData.DataOverlap('yelp_test_set_review.json')

    count = totalData.Preprocess()
    print "count :" + str(count)
    with open(add+'\data2.json', 'w') as outfile:
        list = totalData.data_user.keys()
        for ele in list:
            json.dump(totalData.data_user.get(ele), outfile)
            outfile.write('\n')

main()
