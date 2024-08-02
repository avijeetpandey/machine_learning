# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing dataset
dataset = pd.read_csv("sal.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# traing and spliting the data
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=1 / 3, random_state=0
)

# training simple regression model
regressor = LinearRegression()
regressor.fit(train_x, train_y)

# predicting the value
y_pred = regressor.predict(test_x)

## visualising the data
plt.scatter(train_x,train_y,color="red")
plt.plot(train_x,regressor.predict(train_x), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualsing test set
plt.scatter(test_x, test_y, color = 'red')
plt.plot(train_x, regressor.predict(train_x), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()