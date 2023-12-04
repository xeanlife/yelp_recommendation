## Import Packages
from pyspark import SparkContext, SparkConf
import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from operator import add, itemgetter
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor, Pool


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


## Retrive Attributes
noise_encoder = LabelEncoder().fit(['quiet', 'average', 'loud', 'very_loud']) # encode noise level
def get_attributes(attributes):
  if isinstance(attributes, dict): # if attributes exist
    price = attributes.get('RestaurantsPriceRange2', None) # get price range
    if price:
      price = int(price)
    
    noise_level = attributes.get('NoiseLevel', None) # get noise level
    if noise_level:
      noise_level = noise_encoder.transform([noise_level])[0]
    
    alcohol = attributes.get('Alcohol', None) # get alcohol
    if alcohol == 'none':
      alcohol = 0
    elif alcohol:
      alcohol = 1

    drivethru = attributes.get('DriveThru', None) # get drive through
    if drivethru == 'True':
      drivethru = 1
    elif drivethru == 'False':
      drivethru = 0
    
    kids = attributes.get('GoodForKids', None) # get good for kids
    if kids == 'True':
      kids = 1
    elif kids == 'False':
      kids = 0
    
    return [price, noise_level, alcohol, drivethru, kids]
  else:
    return [None]*5


## Retrive Categories
def prep_cate(categories):
  if isinstance(categories, str):
    return list(categories.split(', '))
  else:
    return []
  
  
## Get Category Encodings
def bus_category(categories):
  cate_list = prep_cate(categories)
  if cate_list: # if categories exist
    cate_list = list(set(cate_list) & set(cate_top))
    return cate_encode.transform([cate_list]).tolist()[0]
  else:
    return [None]*30


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


## New User/Business
yelp_user_ids = yelp_user.map(lambda user_dict: user_dict['user_id']).collect() # users that were found
yelp_business_ids = yelp_business.map(lambda business_dict: business_dict['business_id']).collect() # businesses that were found

new_user = list(set(yelp_test_user) - set(yelp_user_ids)) # users not found
new_business = list(set(yelp_test_business) - set(yelp_business_ids)) # businesses not found

yelp_test_new = yelp_test.filter(lambda user_business: (user_business[0] in new_user) or (user_business[1] in new_business)) # pairs that contain new
yelp_test = yelp_test.filter(lambda user_business: not ((user_business[0] in new_user) or (user_business[1] in new_business))).persist() # regular pairs


## Prepare User Features
yelp_user_dicts = yelp_user.collect() # get all user as a list of dictionaries
yelp_user_dicts_df = pd.DataFrame(yelp_user_dicts) # convert into df
yelp_user_dicts_df = yelp_user_dicts_df.set_index('user_id') # set user id as index
yelp_user_dicts_df_friends = yelp_user_dicts_df.apply(lambda row: row['friends'].split(', ') if row['friends'] != 'None' else [], axis=1) # split friends into list
yelp_user_dicts_df['yelping_since'] = yelp_user_dicts_df['yelping_since'].str[:4].apply(int) # impute to year
yelp_user_dicts_df['friends_cnt'] = yelp_user_dicts_df_friends.apply(len) # get number of friends
yelp_user_dicts_df['elite'] = yelp_user_dicts_df.apply(lambda row: row['elite'].split(', ') if row['elite'] != 'None' else [], axis=1).apply(len) # get number of elites
yelp_user_dicts_df = yelp_user_dicts_df.drop(['name','friends'], axis=1) # remove name, friends


## Prepare Business Features
yelp_business_dicts = yelp_business.collect() # get all business as a list of dictionaries
yelp_business_dicts_df = pd.DataFrame(yelp_business_dicts) # convert into df
yelp_business_dicts_df = yelp_business_dicts_df.set_index('business_id') # set business id as index
yelp_business_dicts_df['postal_code'] = yelp_business_dicts_df['postal_code'].str[0].apply(lambda code: ord(code) if isinstance(code, str) else None) # seperate by first character of zip
cate_top = yelp_business.map(lambda business_dict: prep_cate(business_dict['categories'])).flatMap(lambda x: x).map(lambda x: (x, 1)).reduceByKey(add).sortBy(lambda x: -x[1]).keys().take(30) # get top categories
cate_encode = MultiLabelBinarizer().fit([cate_top]) # encode category
yelp_business_dicts_df = yelp_business_dicts_df.join(pd.DataFrame(yelp_business_dicts_df['categories'].apply(bus_category).tolist(), index=yelp_business_dicts_df.index), rsuffix='_cate') # trasnform category
yelp_business_dicts_df = yelp_business_dicts_df.join(pd.DataFrame(yelp_business_dicts_df['attributes'].apply(get_attributes).tolist(), index=yelp_business_dicts_df.index), rsuffix='_attr') # trasnform attributes
yelp_business_dicts_df['hours'] = yelp_business_dicts_df['hours'].apply(lambda hours: len(hours) if hours else None) # convert hours to days open
yelp_business_dicts_df = yelp_business_dicts_df.drop(['name','neighborhood','address','city','state','latitude','longitude','categories', 'attributes'], axis=1) # remove unneeded columns


## Manage Feature Transformation & Friends
def to_dict(a):
  return {a[0]:a[1]}

def add_dict(a, b):
  a.update({b[0]:b[1]})
  return a

def combine_dict(a, b):
  a.update(b)
  return a


def transform_X(user_id, business_id):
  user_X = yelp_user_dicts_df.loc[user_id].tolist() # get user side X
  business_X = yelp_business_dicts_df.loc[business_id].tolist() # get business side X
  mutual_friends = set(yelp_user_dicts_df_friends[user_id])&yelp_train_bus_user.get(business_id, set()) # is a friend & have rated the business
  if mutual_friends: # if there have mutual friends
    mutual_friends_rate = itemgetter(*mutual_friends)(yelp_train_bus_user_rate[business_id]) # get mutual friends ratings
    friends_X = [np.mean(mutual_friends_rate)] # get mutual friends average rating
  else: # else return none
    friends_X = [None]
  return user_X+business_X+friends_X # return all features


def predict_new(user_id, business_id):
  if user_id in yelp_user_dicts_df.index: # check if user is new
    return yelp_user_dicts_df.loc[user_id]['average_stars'] # get user side average
  elif business_id in yelp_business_dicts_df.index: # check if business is new
    return yelp_business_dicts_df.loc[business_id]['stars'] # get business side average
  else:
    return yelp_train_mean # return all features


yelp_train_bus_user = yelp_train.map(lambda user_business_rate: (user_business_rate[1], user_business_rate[0])).groupByKey().mapValues(set).collectAsMap() # get users of all businesses
yelp_train_bus_user_rate = yelp_train.map(lambda user_business_rate: (user_business_rate[1], (user_business_rate[0], user_business_rate[2]))).combineByKey(to_dict, add_dict, combine_dict).collectAsMap() # get users of all businesses with their ratings
yelp_train_mean = yelp_train.map(lambda user_business_rate: user_business_rate[2]).mean() # get all review mean


## Transform Train
time.sleep(5)
yelp_train_transformed = yelp_train.map(lambda user_business_rate: (transform_X(user_business_rate[0], user_business_rate[1]), user_business_rate[2])).collect() # tranform X
yelp_train_transformed_X = [user_business_rate[0] for user_business_rate in yelp_train_transformed] # extract only features
yelp_train_transformed_y = [user_business_rate[1] for user_business_rate in yelp_train_transformed] # extract only ratings
yelp_train_transformed_X_df = pd.DataFrame(yelp_train_transformed_X) # transform into df
yelp_train_transformed_y_df = pd.DataFrame(yelp_train_transformed_y) # transform into df


## Fit XGBRegressor
xgb_tree_best = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, colsample_bytree=0.3)
xgb_tree_best.fit(yelp_train_transformed_X_df, yelp_train_transformed_y_df)


## Fit Catboost
yelp_train_transformed_X_pool = Pool(yelp_train_transformed_X_df, yelp_train_transformed_y_df)
cat_reg = CatBoostRegressor(n_estimators=800, learning_rate=0.05, depth=10, l2_leaf_reg=10, loss_function='RMSE', silent=True)
cat_reg.fit(yelp_train_transformed_X_pool)


## Print Duration
print('Duration:', time.time() - start_time)


## Save Models
pickle.dump(xgb_tree_best, open('xgb_tree_best.pkl', 'wb'))
pickle.dump(cat_reg, open('cat_reg.pkl', 'wb'))


## Cross Validation
# CatBoostRegressor
yelp_train_transformed_X_pool = Pool(yelp_train_transformed_X_df, yelp_train_transformed_y_df)
cat_reg = CatBoostRegressor(loss_function='RMSE', task_type='GPU', silent=True)
cat_param = {'n_estimators':[600,700,800], 'depth':[6,7,8,10], 'learning_rate':[0.01,0.05,0.1,0.15], 'l2_leaf_reg':[0.1,1,5,10]} # generate parameters to test
cat_grid = GridSearchCV(cat_reg, cat_param, scoring='neg_root_mean_squared_error', verbose=3, cv=5)
cat_grid.fit(yelp_train_transformed_X_df, yelp_train_transformed_y_df) # fit model
cat_grid.best_params_ # {'depth': 10, 'l2_leaf_reg': 10, 'learning_rate': 0.05, 'n_estimators': 800}
cat_grid.best_score_

## XGBRegressor
xgb_tree = xgb.XGBRegressor(tree_method='gpu_hist')
xgb_param = {'n_estimators':[600,700,800], 'max_depth':[6,7,8,10], 'learning_rate':[0.01,0.05,0.1,0.15], 'colsample_bytree':[0.3,0.5,0.7]} # generate parameters to test
xgb_grid = GridSearchCV(xgb_tree, xgb_param, scoring='neg_root_mean_squared_error', verbose=3, n_jobs=5, cv=5)
xgb_grid.fit(yelp_train_transformed_X_df, yelp_train_transformed_y_df) # fit model
xgb_grid.best_params_ # {'colsample_bytree': 0.3, 'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 600}
xgb_grid.best_score_
