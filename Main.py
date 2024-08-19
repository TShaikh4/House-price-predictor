# california house price prediction machine learning model

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# importing dataset
from sklearn.datasets import fetch_california_housing

# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# Display the first few rows
print(df.head())

# pasting the first 5 rows of the dataset (MedInc stands for median income)
# MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  Price
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23  4.526
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22  3.585
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24  3.521
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25  3.413
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25  3.422

# check for missing values: all returned 0 so i'm assuming it's a full dataset (makes sense since this is a training dataset from Sckit-Learn)
# print(df.isnull().sum())

# visualizing the data;just opened a bunch of plot graphs, not really useful atp
# sns.pairplot(df)
# plt.show()

# splitting the data into training and testing sets
X = df.drop('Price', axis=1)
y = df['Price']

# 20% of the data will be used for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluating model with mean squared error; Mean Squared Error: 0.5558915986952437 (lower is better)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# visualizing actuall vs predicted prices
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual vs Predicted Prices')
# plt.show()

# using the model to make predictions; 
# Predicted Price: [4.15194306] (8.3% off of the actual price for these metrics, not bad (House #0 in the dataset))
# Predicted Price: [2.41257139] (29.5% off of the actual price for these metrics, not a great second number (House #1 in the dataset))
new_data = np.array([[3.8462, 53, 6.281853, 1.081081, 565, 2.181467, 37.85, -122.25]])
predicted_price = model.predict(new_data)
print(f'Predicted Price: {predicted_price}')


