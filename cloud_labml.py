import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample data
# Feature matrix
X = np.array([[1], [2], [3], [4], [5]])
# Target values
y = np.array([1, 2, 3, 4, 5])

# Initialize and train the model
RandomForestRegModel = RandomForestRegressor()
RandomForestRegModel.fit(X, y)

# Predict with new data
X_marks = [[70]]
prediction = RandomForestRegModel.predict(X_marks)
print(f"Prediction for {X_marks}: {prediction}")

# Optionally, visualize the result
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, RandomForestRegModel.predict(X), color='red', label='Prediction')
plt.scatter(X_marks, prediction, color='green', label='New Prediction')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Random Forest Regression')
plt.legend()
plt.show()
