import pandas as pd
from sklearn.preprocessing import StandardScaler

# Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def one_hot(original_dataframe, feature_to_encode,drop = False):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], drop_first=drop)
    dummies = dummies.astype(int)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

# https://www.geeksforgeeks.org/encoding-categorical-data-in-sklearn/
def frequency_encode(original_dataframe, feature_to_encode, target):
    frequency_encoded = original_dataframe[feature_to_encode].value_counts(normalize=True)
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].map(frequency_encoded)
    return original_dataframe

def target_encode(original_dataframe, feature_to_encode, target):
    # Calculate the mean of the target for each category in the feature
    target_encoded = original_dataframe.groupby(feature_to_encode)[target].mean()
    
    # Replace the feature with the target-encoded values
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].map(target_encoded)
    
    return original_dataframe

df = pd.read_json('data/raw/loan_approval_dataset.json')

#print(df.head(3).to_markdown(index=False)) #for md file

# Remove Identifying Attributes
data = df.drop('Id',axis=1)

# Frequency Encode - Dont use target because that introduces data leakage, annoying to deal with.
data = frequency_encode(data, 'Profession', 'Risk_Flag')
data = frequency_encode(data, 'CITY', 'Risk_Flag')
data = frequency_encode(data, 'STATE', 'Risk_Flag')

# Standardize numerical features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).drop('Risk_Flag', axis=1).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# One-hot encode
data = one_hot(data, 'Married/Single',True)
data = one_hot(data, 'Car_Ownership',True)
data = one_hot(data, 'House_Ownership', True)

#print(data.head())

# print(data.head(3).to_markdown(index=False)) for md

# Check for na values (none found)
print(data.isna().any())
print(data.isnull().sum())

# print(data.isnull().sum().to_markdown(index=True)) for md

data.to_csv('data/raw/processed_data.csv',index=False)