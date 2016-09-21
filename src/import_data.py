import numpy as np
import pandas as pd
import os.path

#start script

data_folder = r'C:\GitRepository\kaggle-house-prices\data'
path_train = os.path.join(data_folder, 'train.csv')
path_test = os.path.join(data_folder, 'test.csv')

training_data = pd.read_csv(path_train)
test_data = pd.read_csv(path_test)

data_frame = pd.concat((training_data.loc[:,'MSSubClass':'SaleCondition'],
                      test_data.loc[:,'MSSubClass':'SaleCondition']))
       
#log transform skewed numeric features:
training_skew = data_frame.skew(numeric_only = True)
training_skew = training_skew[training_skew > 0.70].index
data_frame[training_skew] = np.log1p(data_frame[training_skew])

#convert categories to numerical values
data_frame = pd.get_dummies(data_frame)
    
#fill in missing data with alternate label
data_frame = data_frame.fillna(data_frame.mean())

#convert features to numpy array
features_train = data_frame[:training_data.shape[0]]
features_test = data_frame[training_data.shape[0]:]

#save data to file for future processing
np.save(os.path.join(data_folder, 'features_train'), np.array(features_train))
np.save(os.path.join(data_folder, 'features_test'), np.array(features_test)) 

training_data["SalePrice"] = np.log1p(training_data["SalePrice"])
y = np.array(training_data.SalePrice)
np.save(os.path.join(data_folder, 'outcome'), y)