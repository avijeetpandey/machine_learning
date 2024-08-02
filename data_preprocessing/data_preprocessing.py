# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Reading data from pandas
dataset = pd.read_csv("Data.csv")

# creating feature matrix
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# printing data sets
print(x)
print(y)

# handling of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])

# update the matrix by the columns
x[:, 1:3] = imputer.transform(x[:, 1:3])

# transforming string labels
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)

x = np.array(ct.fit_transform(x))

# encoding the labels
le = LabelEncoder()
y = le.fit_transform(y)

# training and testing data
x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=0.2, random_state=1)

# feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])