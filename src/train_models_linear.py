import numpy as np
import os.path
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

data_folder = r'C:\GitRepository\HousePrices\data'

X = np.load(os.path.join(data_folder, 'features_train.npy'))
y = np.load(os.path.join(data_folder, 'outcome.npy'))

def calculate_cv_error(model):
    score = cross_val_score(model, X, y, scoring='mean_squared_error', cv=4)
    return np.sqrt(-score).mean()

def find_errors_regularization(Model, alphas):
    error = [calculate_cv_error(Model(alpha = alpha)) for alpha in alphas]
    min_index, min_value = min(enumerate(error), key=lambda p: p[1])
    return min_index, min_value

#linear regression
linear_error = calculate_cv_error(LinearRegression())
print 'Vanilla linear regression minimum error = %f' % linear_error

#linear regression with regularization
alphas = [1, 3, 7, 10, 15, 20, 30]
min_index, min_value = find_errors_regularization(Ridge, alphas)
print 'Ridge minimum error = %f for alpha = %0.1f' % (min_value, alphas[min_index])

#Lasso linear regression 
alphas_ls = [0.0003, 0.0007, 0.001, 0.0015, 0.003, 0.01]
min_index_lasso, min_value_lasso = find_errors_regularization(Lasso, alphas_ls)
print 'Lasso minimum error = %f for alpha = %0.4f'\
        % (min_value_lasso, alphas_ls[min_index_lasso])

#ElasticNet linear Regression
l1_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
elasticnet_error = [calculate_cv_error(ElasticNet(alphas_ls[min_index_lasso], 
                                                  l1_ratio=l1_ratio)) for l1_ratio in l1_ratios]

min_index_elastic, min_value_elastic = min(enumerate(elasticnet_error), key=lambda p: p[1])
print 'ElasticNet minimum error = %f for alpha = %0.4f, l1_ratio = %0.2f'\
        % (min_value_elastic, alphas_ls[min_index_lasso], l1_ratios[min_index_elastic])
