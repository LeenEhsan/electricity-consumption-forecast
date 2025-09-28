import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy training data
X = np.array([
    [25, 60, 4, 1012, 20, 39],
    [30, 55, 3, 1010, 25, 38],
    [22, 65, 2, 1008, 10, 40],
    [35, 45, 5, 1015, 30, 41],
    [28, 58, 3, 1009, 22, 37]
])
y = np.array([67000, 69000, 65000, 72000, 68000])

# Train and save the model
model = LinearRegression()
model.fit(X, y)

with open("power_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")


