## Import Packages
from pyspark import SparkContext, SparkConf
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from operator import add
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


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
xgb_tree = LinearRegression(n_jobs=-1)
null_imputer = SimpleImputer().fit(yelp_train_transformed_X_df)
yelp_train_transformed_X_df = null_imputer.transform(yelp_train_transformed_X_df)
xgb_tree.fit(yelp_train_transformed_X_df, yelp_train_transformed_y_df)


## Transform Test
yelp_test_transformed = yelp_test.map(lambda user_business: ((yelp_user_dict.get(user_business[0], user_dict_default), yelp_business_dict.get(user_business[1], business_dict_default)), user_business)).collect()
yelp_test_transformed_X = [user_business[0][0]+user_business[0][1] for user_business in yelp_test_transformed] # extract only features
yelp_test_transformed_og = [user_business[1] for user_business in yelp_test_transformed] # extract only original names
yelp_test_transformed_X_df = pd.DataFrame(yelp_test_transformed_X) # transform into df
yelp_test_result = pd.DataFrame(yelp_test_transformed_og, columns=['user_id', 'business_id']) # transform into df


## Predict
yelp_test_transformed_X_df = null_imputer.transform(yelp_test_transformed_X_df)
xgb_predicted = xgb_tree.predict(yelp_test_transformed_X_df)
yelp_test_result['prediction'] = xgb_predicted # combine result and original 


## Print Duration
print('Duration:', time.time() - start_time)


## Write To File
yelp_test_result.to_csv(output_file_name, index=False)

