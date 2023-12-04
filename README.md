# Yelp Restaurant Recommendation Project

## Objective
To build different types of recommendation systems using the yelp training data to predict the ratings/stars for given user ids and business ids. You can make any improvement to your recommendation system in terms of speed and accuracy.

## Models Used
[Item-based CF](./competition_item_based.py)
[XGBoost + Catboost](./competition.py)
[Catboost Regression](./competition_regression.py)
[Linear Regression](./competition_linear.py)

## Final Results
The [final model](./competition.py) was mixed using XGBoost, Catboost, and CF with user friends, then refined with grid-search on many different parameters.
Train RMSE: 0.9722, Validation RMSE: 0.9742, Test RMSE: Ranked #3 out of 300 (numbers undisclosed).
