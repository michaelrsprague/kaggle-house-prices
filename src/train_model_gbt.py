import numpy as np
import os.path
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

data_folder = r'C:\GitRepository\kaggle-house-prices\data'

X = np.load(os.path.join(data_folder, 'features_train.npy'))
y = np.load(os.path.join(data_folder, 'outcome.npy'))

#best is 200
search_parameters = {'max_features':[80, 120, 200], 
                     'min_samples_split': [2, 3],
                     'subsample': [0.8, 0.9, 1.0]}

model_base = GradientBoostingRegressor(n_estimators = 200, learning_rate = 0.1, max_depth = 3)

clf = GridSearchCV(model_base, search_parameters, scoring='mean_squared_error', cv = 5)
clf.fit(X, y)

print clf.best_params_, np.sqrt(-clf.best_score_)
