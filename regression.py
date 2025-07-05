from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import load_data, split_data, evaluate_model

# Load data
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# -----------------------
# Model 1: Linear Regression (no tuning)
lr = LinearRegression()
lr.fit(X_train, y_train)
mse_lr, r2_lr = evaluate_model(lr, X_test, y_test)
print("Linear Regression -> MSE: {:.2f}, R2: {:.2f}".format(mse_lr, r2_lr))

# -----------------------
# Model 2: Decision Tree (tuned)
dt_params = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5, scoring='neg_mean_squared_error')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
mse_dt, r2_dt = evaluate_model(best_dt, X_test, y_test)
print("Decision Tree Best Params:", grid_dt.best_params_)
print("Decision Tree -> MSE: {:.2f}, R2: {:.2f}".format(mse_dt, r2_dt))

# -----------------------
# Model 3: Random Forest (tuned)
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_squared_error')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
mse_rf, r2_rf = evaluate_model(best_rf, X_test, y_test)
print("Random Forest Best Params:", grid_rf.best_params_)
print("Random Forest -> MSE: {:.2f}, R2: {:.2f}".format(mse_rf, r2_rf))