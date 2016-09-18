#created an implementation of pd.get_dummies for educational purposes

def find_column_dictionary(column):
    
    dictionary = {}
    value = 0
    
    for entry in column:
        
        if entry not in dictionary:
            dictionary[entry] = value
            value += 1        
        
    return dictionary

#convert non-numeric categories into numeric categories
def convert_column_numerical(data_frame, column):
    
    dictionary = find_column_dictionary(data_frame[column])
    data_frame[column].replace(dictionary, inplace = True)
    return data_frame

def convert_category_numerical(training_data):

    for column in training_data.columns:     
    
        if training_data[column].dtype == object:
            training_data = convert_column_numerical(training_data, column)