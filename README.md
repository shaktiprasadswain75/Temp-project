# Weather Prediction for 2 Days using ML
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'day': list(range(1, 11)),  # last 10 days
    'temperature': [30, 32, 34, 33, 35, 36, 37, 36, 38, 39]
}
df = pd.DataFrame(data)


X = df[['day']]
y = df['temperature']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

next_days = [[11], [12]]  # Predict for next 2 days
predicted_temp = model.predict(next_days)

print("Predicted Temperature for Next 2 Days:")
print(f"Day 11: {predicted_temp[0]:.2f}°C")
print(f"Day 12: {predicted_temp[1]:.2f}°C")


y_pred = model.predict(X_test)
print("\nModel Accuracy (R²):", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted Line')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weather Prediction')
plt.legend()
plt.show()
