from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import encode_binary, encode_onehot, encode_ordinal, split_dataframes

adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features
y = adult.data.targets 

X = X.drop(['fnlwgt', 'education-num'],axis=1)

y = y[~X.isna().any(axis=1)]
X = X[~X.isna().any(axis=1)]

columns = list(X.columns)

encoding_type = [None,
                 "onehot",
                 "ordinal",
                 "onehot",
                 "onehot",
                 "onehot",
                 "onehot",
                 "binary",
                 None,
                 None,
                 None,
                 "onehot"]

column_names = X.columns

for column, encoding in zip(column_names, encoding_type):
    if encoding == "onehot":
        X = encode_onehot(X, column)
    elif encoding == "ordinal":
        X = encode_ordinal(X, column)
    elif encoding == "binary":
        X = encode_binary(X, column)
    elif encoding is None:
        continue  # Do nothing for 'None'

X["bias"] = 1
y = y.replace("<=50K.", "<=50K")
y = y.replace(">50K.", ">50K")
y = encode_binary(y, "income")

train_X, val_X, test_X, train_y, val_y, test_y = split_dataframes(X, y, seed=1)

train_X = train_X.to_numpy().astype(np.float32)
val_X = val_X.to_numpy().astype(np.float32)
test_X = test_X.to_numpy().astype(np.float32)
train_y = train_y.to_numpy().astype(np.float32)
val_y = val_y.to_numpy().astype(np.float32)
test_y = test_y.to_numpy().astype(np.float32)

cols  = ["age", "capital-gain", "capital-loss", "hours-per-week"]

print(train_X.shape)
for col in cols:
    train_X = train_X / train_X.max(axis=0)
    val_X = val_X / val_X.max(axis=0)
