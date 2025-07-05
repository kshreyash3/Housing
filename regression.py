import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from utils import load_data, split_data, evaluate_model

# Load and split
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
mse_lr, r2_lr = evaluate_model(lr, X_test, y_test)
print("Linear Regression -> MSE: {:.2f}, R2: {:.2f}".format(mse_lr, r2_lr))

# Model 2: Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
mse_dt, r2_dt = evaluate_model(dt, X_test, y_test)
print("Decision Tree -> MSE: {:.2f}, R2: {:.2f}".format(mse_dt, r2_dt))

# Model 3: Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
mse_rf, r2_rf = evaluate_model(rf, X_test, y_test)
print("Random Forest -> MSE: {:.2f}, R2: {:.2f}".format(mse_rf, r2_rf))

