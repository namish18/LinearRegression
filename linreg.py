# All the required libraries are imported first
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# The data is loaded from URL
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
df = pd.DataFrame(data)


# Scatter plot using Matplotlib
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()

# Data is split into training and testing dataset
X = df[['Hours']]
y = df['Scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# Make Prediction
y_pred = model.predict(X_test)


# Final Results
# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R-squared value
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared value:", r2)

# Plotting the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Hours vs Scores (with Regression Line)')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()

# Predict the score for 9.25 hours of study
predicted_score = model.predict([[9.25]])
predicted_score[0]
