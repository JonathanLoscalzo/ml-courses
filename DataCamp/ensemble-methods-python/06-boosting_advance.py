
# FLAVORS
# Extreme Gradient Boosting XGBoost
# Light Gradient Boosting Machine LGBM
# Categorical Boosting CatBoost


# Movie revenue prediction with CatBoost
# Let's finish up this chapter on boosting by returning to the movies dataset! In this exercise, you'll build a CatBoostRegressor to predict the log-revenue. Remember that our best model so far is the AdaBoost model with a RMSE of 5.15.

# Will CatBoost beat AdaBoost? We'll try to use a similar set of parameters to have a fair comparison.

# Recall that these are the features we have used so far: 'budget', 'popularity', 'runtime', 'vote_average', and 'vote_count'. catboost has been imported for you as cb.


# Build and fit a CatBoost regressor
reg_cat = cb.CatBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=500)
reg_cat.fit(X_train, y_train)

# Calculate the predictions on the set set
pred = reg_cat.predict(X_test)

# Evaluate the performance using the RMSE
rmse_cat = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE (CatBoost): {:.3f}'.format(rmse_cat))
# RMSE (CatBoost): 5.110

# Build and fit a XGBoost regressor
reg_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=500)
reg_xgb.fit(X_train, y_train)

# Build and fit a LightGBM regressor
reg_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, seed=500)
reg_lgb.fit(X_train, y_train)

# Calculate the predictions and evaluate both regressors
pred_xgb = reg_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
pred_lgb = reg_lgb.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

print('Extreme: {:.3f}, Light: {:.3f}'.format(rmse_xgb, rmse_lgb))
## Extreme: 5.122, Light: 5.142