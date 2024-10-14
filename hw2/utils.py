import pandas as pd
import numpy as np

education_levels = [[
	    "Preschool",
		"1st-4th",
		"5th-6th",
		"7th-8th",
		"9th",
		"10th",
		"11th",
		"12th",
		"HS-grad",
		"Some-college",
		"Assoc-voc",
		"Assoc-acdm",
		"Bachelors",
		"Masters",
		"Doctorate",
		"Prof-school"
]]

def split_dataframes(X, y, train_frac=0.8, validate_frac=0.1, seed=None):
    """ Split DataFrame and target DataFrame into train, validate, and test sets """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle the DataFrame rows
    permuted_indices = np.random.permutation(X.index)
    total_count = len(X)
    
    # Calculate split indices
    train_end = int(train_frac * total_count)
    validate_end = int(validate_frac * total_count) + train_end
    
    # Split the data
    train_X = X.loc[permuted_indices[:train_end]]
    validate_X = X.loc[permuted_indices[train_end:validate_end]]
    test_X = X.loc[permuted_indices[validate_end:]]
    
    # Split the target DataFrame y
    train_y = y.loc[permuted_indices[:train_end]]
    validate_y = y.loc[permuted_indices[train_end:validate_end]]
    test_y = y.loc[permuted_indices[validate_end:]]

    return train_X, validate_X, test_X, train_y, validate_y, test_y

def encode_onehot(df, column):
    """ One-hot encode a column with dummy variables """
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df

def encode_ordinal(df, column):
    """ Encode a column as ordinal """
    # unique_vals = sorted(df[column].unique())
    unique_vals = education_levels[0]
    ordinal_map = {val: i for i, val in enumerate(unique_vals)}
    df[column] = df[column].map(ordinal_map)
    return df

def encode_binary(df, column):
    """ Encode a binary column """
    unique_vals = sorted(df[column].unique())
    print(unique_vals)
    if len(unique_vals) != 2:
        raise ValueError("Column is not binary")
    binary_map = {unique_vals[0]: 0, unique_vals[1]: 1}
    df[column] = df[column].map(binary_map)
    return df
