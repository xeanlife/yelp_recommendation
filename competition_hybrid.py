## Import Packages
from pyspark import SparkContext, SparkConf
import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from copy import deepcopy
from statistics import mean
from operator import add
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


## Read User Input
# folder_path = sys.argv[1] # get folder path from CLI
folder_path = 'compdata/'
user_file_name = folder_path+'user.json' # user json file
business_file_name = folder_path+'business.json' # business json file
train_file_name = folder_path+'yelp_train.csv' # train file
# test_file_name = sys.argv[2] # get test file path from CLI
test_file_name = 'yelp_val_in.csv'
# output_file_name = sys.argv[3] # get output file path from CLI
output_file_name = 'comp.csv'


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


## Retrive Price Range
def price_range(attributes):
  if isinstance(attributes, dict):
    price = attributes.get('RestaurantsPriceRange2', 2)
    return int(price)
  else:
    return None
  
  
## Retrive Noise Level
def noise_level(attributes):
  if isinstance(attributes, dict):
    return noise_encoder.transform([attributes.get('NoiseLevel', 'average')])[0]
  else:
    return None
  

## Retrive Categories
def prep_cate(categories):
  if isinstance(categories, str):
    return list(categories.split(', '))
  else:
    return []


## Read File
start_time = time.time() # start timer
yelp_train = sc.textFile(train_file_name) # read train file
yelp_train = yelp_train.map(lambda line: line.split(',')) # split by comma
header = yelp_train.first()
yelp_train = yelp_train.filter(lambda line: line != header).map(lambda line: (line[0], line[1], float(line[2]))) # remove header and cast star into float

yelp_test = sc.textFile(test_file_name).map(lambda line: line.split(',')) # read test file and split by comma
yelp_test = yelp_test.filter(lambda line: line != header).map(lambda line: (line[0], line[1])) #  remove header and empty stars

yelp_user = sc.textFile(user_file_name).map(json.loads) # read user file
yelp_business = sc.textFile(business_file_name).map(json.loads) # read business file


## Filter Files/Features Reduce Size
yelp_test_user = yelp_test.map(lambda user_business: user_business[0]).distinct().collect() # all users that need to be tested
yelp_test_business = yelp_test.map(lambda user_business: user_business[1]).distinct().collect() # all businesses that need to be tested
yelp_train_user = yelp_train.map(lambda user_business: user_business[0]).distinct().collect() # all users that need to be trained
yelp_train_business = yelp_train.map(lambda user_business: user_business[1]).distinct().collect() # all businesses that need to be trained
yelp_possible_user = list(set(yelp_test_user + yelp_train_user))
yelp_possible_business = list(set(yelp_test_business + yelp_train_business))

yelp_user = yelp_user.filter(lambda user_dict: user_dict['user_id'] in yelp_possible_user) # only care about users that take part in test and train
yelp_business = yelp_business.filter(lambda business_dict: business_dict['business_id'] in yelp_possible_business) # only care about business that take part in test and train
cate_encode = MultiLabelBinarizer().fit([yelp_business.map(lambda business_dict: prep_cate(business_dict['categories'])).flatMap(lambda x: x).map(lambda x: (x, 1)).reduceByKey(add).sortBy(lambda x: -x[1]).keys().take(30)]) # encode category
state_encoder = LabelEncoder().fit(yelp_business.map(lambda business_dict: business_dict['state']).distinct().collect()) # encode state
noise_encoder = LabelEncoder().fit(['quiet', 'average', 'loud', 'very_loud']) # encode noise level

yelp_user = yelp_user.map(lambda user_dict: (user_dict['user_id'], (user_dict['review_count'], user_dict['average_stars'], user_dict['useful']))) # transform into user:review_count,average_stars,useful
yelp_business = yelp_business.map(lambda business_dict: (business_dict['business_id'], (business_dict['stars'], business_dict['review_count'], price_range(business_dict['attributes']), state_encoder.transform([business_dict['state']])[0], noise_level(business_dict['attributes'])) + tuple(cate_encode.transform([prep_cate(business_dict['categories'])]).tolist()[0]))) # transform into business:stars,review_count,price_range,state,noise,category
yelp_user_dict = yelp_user.collectAsMap() # collect for replacement later
yelp_business_dict = yelp_business.collectAsMap() # collect for replacement later


# Set Defaults
user_dict_default = (0,2.5,0)
business_dict_default = (2.5,0,None,None,None) + tuple(cate_encode.transform([[]]).tolist()[0])


## Transform Train
yelp_train_transformed = yelp_train.map(lambda user_business_rate: ((yelp_user_dict.get(user_business_rate[0], user_dict_default), yelp_business_dict.get(user_business_rate[1], business_dict_default)), user_business_rate[2])).collect()
yelp_train_transformed_X = [user_business_rate[0][0]+user_business_rate[0][1] for user_business_rate in yelp_train_transformed] # extract only features
yelp_train_transformed_y = [user_business_rate[1] for user_business_rate in yelp_train_transformed] # extract only ratings
yelp_train_transformed_X_df = pd.DataFrame(yelp_train_transformed_X) # transform into df
yelp_train_transformed_y_df = pd.DataFrame(yelp_train_transformed_y, columns=['stars']) # transform into df


## Fit XGBRegressor
xgb_tree = xgb.XGBRegressor(booster='gbtree')
xgb_tree.fit(yelp_train_transformed_X_df, yelp_train_transformed_y_df)


## Transform Test
yelp_test_transformed = yelp_test.map(lambda user_business: ((yelp_user_dict.get(user_business[0], user_dict_default), yelp_business_dict.get(user_business[1], business_dict_default)), user_business)).collect()
yelp_test_transformed_X = [user_business[0][0]+user_business[0][1] for user_business in yelp_test_transformed] # extract only features
yelp_test_transformed_og = [user_business[1] for user_business in yelp_test_transformed] # extract only original names
yelp_test_transformed_X_df = pd.DataFrame(yelp_test_transformed_X) # transform into df
yelp_test_result = pd.DataFrame(yelp_test_transformed_og, columns=['user_id', 'business_id']) # transform into df


## Predict
xgb_predicted = xgb_tree.predict(yelp_test_transformed_X_df)
yelp_test_result['prediction'] = xgb_predicted # combine result and original


## Read File
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
yelp_test_result_cf = yelp_test.collect() # get result
yelp_test_result_cf_df = pd.DataFrame(yelp_test_result_cf, columns=['user_id', 'business_id', 'prediction_cf']) # cf prediction into df


## Merge Perdictions
yelp_train_business_average_df = pd.DataFrame(yelp_train_business_average.items(), columns=['business_id', 'avg_rating'])
yelp_test_result_combined = yelp_test_result.merge(yelp_train_business_average_df, how='left', left_on='business_id', right_on='business_id') # merge two predictions
yelp_test_result_combined['prediction'] = 0.9*yelp_test_result_combined['prediction'] + 0.1*yelp_test_result_combined['avg_rating']
yelp_test_result_combined = yelp_test_result_combined.drop('avg_rating', axis=1)


## Print Duration
print('Duration:', time.time() - start_time)


## Write To File
yelp_test_result_combined.to_csv(output_file_name, index=False)

