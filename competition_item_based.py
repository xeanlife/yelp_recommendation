## Import Packages
from pyspark import SparkContext, SparkConf
import os
import sys
import csv
import math
import time
import random
from statistics import mean


## Read User Input
# train_file_name = sys.argv[1] # get train file path from CLI
train_file_name = 'yelp_train.csv'
# test_file_name = sys.argv[2] # get test file path from CLI
test_file_name = 'yelp_val_in.csv'
# output_file_name = sys.argv[3] # get output file path from CLI
output_file_name = 't2.csv'


## Set/Initiate Pyshark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
conf = SparkConf().setMaster('local[*]').set('spark.executor.memory', '4g').set('spark.driver.memory', '4g')
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel('WARN')


## Pearson Correlation - Item All Rated
def pearson_cor(test_business, test_user_rating, business_rated_avg, test_business_avg):
  business_rated_sim = business_rated_avg # store result
  for business_rated in business_rated_avg: # go through each rated business of test_user
    business_rated_avg_local = business_rated_avg[business_rated] # get current rated business
    rated_user_rating = dict(yelp_train_business_user_dict[business_rated]) # get current rated business
    # find corated users
    corated_user = set(test_user_rating.keys()) & set(rated_user_rating.keys()) # users that rated both test_business and current rated business
    # rating for corated users
    diff_test = []
    diff_rated = []
    for user in corated_user: # go through each corated user
      diff_test.append(test_user_rating[user] - test_business_avg) # all diff of test business
      diff_rated.append(rated_user_rating[user] - business_rated_avg_local) # all diff of rated business
    numerator = sum(diff_pair[0] * diff_pair[1] for diff_pair in zip(diff_test, diff_rated)) # calcualte numerator
    denominator = math.sqrt(sum([x**2 for x in diff_test]))*math.sqrt(sum([x**2 for x in diff_rated])) # calculate denominator
    if denominator == 0: # gate for dividing by 0, update result
      if numerator == 0: # user rating are all same as average
        business_rated_sim[business_rated] = 1
      else:
        business_rated_sim[business_rated] = 0
    else:
      business_rated_sim[business_rated] = numerator/denominator
  return business_rated_sim


## Item Based CF
def item_cf(test_pair):
  test_user = test_pair[0]
  test_business = test_pair[1]
  business_rating = yelp_train_user_business_dict.get(test_user) # all business,rating of test_user
  test_user_rating = yelp_train_business_user_dict.get(test_business) # all user,rating of test_business
  
  if (not business_rating) and (not test_user_rating): # cold both
    return test_pair + (2.5,)
  elif (not business_rating): # cold user
    test_user_rating = dict(test_user_rating)
    return test_pair + (mean(test_user_rating.values()),) # return avg of business
  elif (not test_user_rating): # cold business
    business_rating = dict(business_rating)
    return test_pair + (mean(business_rating.values()),) # return avg of user
  else: # reformat into dictionary
    business_rating = dict(business_rating)
    test_user_rating = dict(test_user_rating)
  
  business_rated = list(business_rating.keys()) # all businesses test_user rated
  business_rated_avg = {bs:yelp_train_business_average[bs] for bs in business_rated} # business:avg_rating of all businesses test_user rated
  test_business_avg = yelp_train_business_average[test_business] # avg rating for test business
  business_rated_sim = pearson_cor(test_business, test_user_rating, business_rated_avg, test_business_avg) # get all pearson similarities
  business_rated_sim = {k: v for k, v in business_rated_sim.items() if v>0} # only positive similarities
  # {'I1': 0.7657048647896112, 'I3': -0.5854905538443586, 'I4': -0.11043152607484649}
  
  business_rated = list(business_rated_sim.keys()) # obtain new business key
  numerator = 0
  denominator = 0
  for business in business_rated: # calculate final prediction
    numerator += business_rating[business]*business_rated_sim[business] # add to numerator
    denominator += abs(business_rated_sim[business]) # add to denominator
  if denominator == 0: # all similarities are 0
    return test_pair + (mean(business_rating.values()),) # treat as cold business
  else:
    return test_pair + (numerator/denominator,)
# pearson_cor('I2', {'U1':1, 'U3':4, 'U4':3}, {'I1':10/3, 'I3':8/3, 'I4':8/3}, 8/3, {'I1': [('U1', 2), ('U2', 3), ('U4', 5)], 'I3': [('U2', 5), ('U3', 2), ('U4', 1)], 'I4': [('U1', 3), ('U2', 2), ('U3', 3)]})


## Read File
start_time = time.time() # start timer
yelp_train = sc.textFile(train_file_name) # read file
yelp_train = yelp_train.map(lambda line: line.split(',')) # split by comma
header = yelp_train.first()
yelp_train = yelp_train.filter(lambda line: line != header) # remove header
yelp_train_business_user = yelp_train.map(lambda line: [line[1], (line[0], float(line[2]))]).groupByKey().mapValues(list) # reformat to bussiness:(user:rating) and group on business id
yelp_train_user_business = yelp_train.map(lambda line: [line[0], (line[1], float(line[2]))]).groupByKey().mapValues(list) # reformat to user:(bussiness:rating) and group on user id
yelp_train_business_user_dict = yelp_train_business_user.collectAsMap()
yelp_train_user_business_dict = yelp_train_user_business.collectAsMap()

yelp_test = sc.textFile(test_file_name).map(lambda line: line.split(',')) # read test file and split by comma
yelp_test = yelp_test.filter(lambda line: line != header).map(lambda line: (line[0], line[1])) #  remove header and empty stars


## Prediction Calculation
yelp_train_business_average = yelp_train_business_user.map(lambda business_user: ((business_user[0], mean([user_rating[1] for user_rating in business_user[1]])))).collectAsMap() # average for each business
yelp_test = yelp_test.map(item_cf) # do item cf on all test pairs
yelp_test_results = yelp_test.collect() # get result

# yelp_test_pairs = yelp_test.collect()
# yelp_test_results = []
# for yelp_test in yelp_test_pairs:
#   cf = item_cf(yelp_test)
#   # print(yelp_test, cf)
#   yelp_test_results.append(yelp_test + (cf,))


## Print Duration
print('Duration:', time.time() - start_time)

  
## Write To File
with open(output_file_name, 'w', newline='') as f_out:
  writer = csv.writer(f_out)
  _ = writer.writerow(['user_id', 'business_id', 'prediction'])
  _ = writer.writerows(yelp_test_results)
  
