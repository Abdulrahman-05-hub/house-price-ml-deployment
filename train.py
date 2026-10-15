from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

X = np.array([
    [800, 2, 15],
    [1000, 2, 10],
    [1200, 3, 8],
    [1500, 3, 5],
    [1800, 4, 4],
    [2000, 4, 3],
    [2400, 4, 2],
    [2800, 5, 1],
    [900, 2, 20],
    [1600, 3, 12],
    [2200, 4, 7],
    [1300, 3, 15],
])

y = np.array([
    120000, 150000, 190000, 240000,
    300000, 340000, 420000, 500000,
    110000, 220000, 360000, 175000,
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean absolute error on demo data: ${mae:,.2f}")

joblib.dump(model, "house_price_model.joblib")
print("Model saved to house_price_model.joblib")
