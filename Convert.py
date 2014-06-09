__author__ = 'Nikhil'

import json
import pandas as pd
from glob import glob
import AppendUser



def convert(x):
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


print "Start"
PATH = 'Data'
users = AppendUser.AppendUser()
users.initialize(PATH+'\yelp_training_set_review.json',PATH+'\yelp_training_set_user.json')
users.UpdateUser(PATH)

for json_filename in glob(PATH+'\*.json'):
    csv_filename = '%s.csv' % json_filename[:-5]
    print 'Converting %s to %s' % (json_filename, csv_filename)
    datafile = pd.DataFrame([convert(line) for line in file(json_filename)])
    datafile.to_csv(csv_filename, encoding='utf-8', index=False)

