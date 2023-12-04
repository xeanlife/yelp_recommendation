import pandas as pd

pred = pd.read_csv('comp.csv')
truth = pd.read_csv('compdata/yelp_val.csv')

truth_pred = truth.merge(pred, on=['user_id', 'business_id'])
truth_pred['diff'] = truth_pred['prediction']-truth_pred['stars']

truth_pred[truth_pred['diff']>0]['diff'].sum()
truth_pred[truth_pred['diff']>0]['diff'].count()
#####################################################
truth_pred[truth_pred['diff']<0]['diff'].sum()
truth_pred[truth_pred['diff']<0]['diff'].count()

msre = (((truth_pred['prediction']-truth_pred['stars'])**2).mean())**.5
(truth_pred['prediction']-truth_pred['stars']).abs().value_counts(bins=[0,1,2,3,4])


print('MSE:', msre)


# XGBRegressor (user:review_count,average_stars,useful; business:stars,review_count,price_range,state,noise,category): 0.9826, 54763, -54531
# XGBRegressor (all + friends, n_estimators=800, learning_rate=0.05, max_depth=6, colsample_bytree=0.3): 0.97511, 54053, -53781
# CatBoostRegressor (all + friends, n_estimators=700, learning_rate=0.05, depth=10, loss_function='RMSE'): 0.97481, 54026, -53797
# XGBRegressor + CatBoostRegressor: 0.97421, 53985, -53746