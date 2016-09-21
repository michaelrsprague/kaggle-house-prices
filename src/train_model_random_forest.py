import numpy as np
import os.path
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor

data_folder = r'C:\GitRepository\kaggle-house-prices\data'

X = np.load(os.path.join(data_folder, 'features_train.npy'))
y = np.load(os.path.join(data_folder, 'outcome.npy'))

def calculate_cv_error(model):
    score = cross_val_score(model, X, y, scoring='mean_squared_error', cv=5)
    return np.sqrt(-score).mean()

n_estimators_quick = 50
n_estimators_slow = 150

max_feats = [40, 80, 100, 140, 200]
error = [calculate_cv_error(RandomForestRegressor(n_estimators = n_estimators_quick, min_samples_leaf = 2,
         max_features=max_feat)) for max_feat in max_feats]

min_index, min_value = min(enumerate(error), key=lambda p: p[1])

print "Best results for max_features = %d" %(max_feats[min_index])

print "Minimum RMSE error = %0.4f for n_estimators = %d" % \
        (calculate_cv_error(RandomForestRegressor(
            n_estimators = n_estimators_slow, min_samples_leaf = 2, max_features=max_feats[min_index])),
         n_estimators_slow)
