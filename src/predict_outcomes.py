import numpy as np
import os.path
import pandas as pd
from sklearn.linear_model import ElasticNet

data_folder = r'C:\GitRepository\HousePrices\data'
path_test = os.path.join(data_folder, 'test.csv')
test_data = pd.read_csv(path_test)

X_train = np.load(os.path.join(data_folder, 'features_train.npy'))
X_test = np.load(os.path.join(data_folder, 'features_test.npy'))
y = np.load(os.path.join(data_folder, 'outcome.npy'))

#fit model and make prediction
model = ElasticNet(0.0007, 0.70).fit(X_train, y)
prediction = np.expm1(model.predict(X_test))
solution = pd.DataFrame({"SalePrice":prediction, "id":test_data.Id})

#switch column order
col = solution.columns.tolist()
col = col[-1:] + col[:-1]
solution[col].to_csv("prediction.csv", index=False)
