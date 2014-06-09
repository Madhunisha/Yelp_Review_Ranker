__author__ = 'Nikhil'

import json


class AppendUser:
    def __init__(self):
        self.data_review = []
        self.data_user = {}

    def initialize(self,reviewFile,userFile):
        cnt = 0
        with open(reviewFile) as f:
            for line in f:
                ind = json.loads(line)
                self.data_review.append(ind)
                cnt += 1

        with open(userFile) as f:
            for line in f:
                ind = json.loads(line)
                ind["status"] = "old"
                self.data_user[ind['user_id']] = ind


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


    def UpdateUser(self,add):
        self.Preprocess()
        with open(add +'\yelp_training_set_user.json', 'w') as outfile:
            list = self.data_user.keys()
            for ele in list:
                json.dump(self.data_user.get(ele), outfile)
                outfile.write('\n')