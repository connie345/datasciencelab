import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from scipy.stats import skew
from scipy.stats import pearsonr

from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

# Import the data
test = pd.read_csv("HousePriceData/test.csv",delimiter=',')
train = pd.read_csv("HousePriceData/train.csv", delimiter=',')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

# Data preprocessing

# Log transform the sale price
train["SalePrice"] = np.log1p(train["SalePrice"])

# Log transform skewed numeric values
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))


skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Get dummies variables for data

all_data = pd.get_dummies(all_data)

# Fill NaN with mean data
all_data = all_data.fillna(all_data.mean()) # Chane this to trian mean.

#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# Creating model

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Part 2

#Creating model and running prediction
model_ridge = Ridge(alpha=0.1).fit(X_train,y)
first_pred = np.expm1(model_ridge.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":first_pred})
solution.to_csv("first_pred.csv", index = False)

# Part 3

alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

print("Best Ridge: %f" % min(cv_ridge))

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.savefig("Ridge_MSE_Alpha.png")
plt.cla()


alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1]

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
plt.plot(alphas,cv_lasso)
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.savefig("Lasso_MSE_Alpha.png")

print("Best Lasoo: %f" % min(cv_lasso))

# Part 4

alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1]

non_zero = [sum(Lasso(alpha=alpha).fit(X_train,y).coef_ != 0) for alpha in alphas]
plt.plot(alphas,non_zero)
plt.xlabel("alpha")
plt.ylabel("non-zero coefs")
plt.savefig("Lasso_NZCoef_Alpha.png")

# Part 5

model_lasso = Lasso(alpha=0.005).fit(X_train,y)
model_lasso_data = model_lasso.predict(X_train)
model_lasso_data_test = model_lasso.predict(X_test)
X_train_temp = np.column_stack((X_train,model_lasso_data))
X_test_temp = np.column_stack((X_test,model_lasso_data_test))

model_ridge = Ridge(alpha=9).fit(X_train,y)
model_ridge_data = model_ridge.predict(X_train)
model_ridge_data_test = model_ridge.predict(X_test)
X_train_temp = np.column_stack((X_train_temp,model_ridge_data))
X_test_temp = np.column_stack((X_test_temp,model_ridge_data_test))

def rmse_cv2(model):
    rmse= np.sqrt(-cross_val_score(model, X_train_temp, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
plt.plot(alphas,cv_ridge)
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.savefig("Ridge_MSE_2_Alpha.png")

print("Best Ridge: %f" % min(cv_ridge))


model_ridge = Ridge(alpha=9).fit(X_train_temp,y)

ridge_preds = np.expm1(model_ridge.predict(X_test_temp))

solution = pd.DataFrame({"id":test.Id, "SalePrice":ridge_preds})
solution.to_csv("ridge_stacking_sol.csv", index = False)

# Part 6

dtrain = xgb.DMatrix(X_train,label=y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
plt.savefig("XGB_Boost.png")

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))

solution = pd.DataFrame({"id":test.Id, "SalePrice":xgb_preds})
solution.to_csv("xgb_sol.csv", index = False)

# Part 7

# We got the best score when feature stacking and then running a ridge regression.

# We can start by feature stacking with a XGB boost

model_lasso = Lasso(alpha=0.005).fit(X_train,y)
model_lasso_data = model_lasso.predict(X_train)
model_lasso_data_test = model_lasso.predict(X_test)
X_train_temp = np.column_stack((X_train,model_lasso_data))
X_test_temp = np.column_stack((X_test,model_lasso_data_test))

model_ridge = Ridge(alpha=9).fit(X_train,y)
model_ridge_data = model_ridge.predict(X_train)
model_ridge_data_test = model_ridge.predict(X_test)
X_train_temp = np.column_stack((X_train_temp,model_ridge_data))
X_test_temp = np.column_stack((X_test_temp,model_ridge_data_test))

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
xgb_preds_train = np.expm1(model_xgb.predict(X_train))
xgb_preds_test = np.expm1(model_xgb.predict(X_test))
X_train_temp = np.column_stack((X_train_temp,xgb_preds_train))
X_test_temp = np.column_stack((X_test_temp,xgb_preds_test))

def rmse_cv3(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15, 30, 50, 75]
# cv_ridge = [rmse_cv3(Ridge(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_ridge)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.savefig("Ridge_MSE_3_Alpha.png")
# plt.show()
# print("Best Ridge: %f" % min(cv_ridge))

# alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15, 30, 50, 75]
# cv_lasso = [rmse_cv3(Lasso(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_lasso)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.savefig("Lasso_MSE_2_Alpha.png")
# plt.show()
# print("Best Ridge: %f" % min(cv_lasso))

model = Ridge(alpha=8).fit(X_train_temp,y)
model_preds = np.expm1(model.predict(X_test_temp))

solution = pd.DataFrame({"id":test.Id, "SalePrice":model_preds})
solution.to_csv("problem7_attempt1_sol.csv", index = False)

# Second Attempt 

model_lasso = Lasso(alpha=0.005).fit(X_train,y)
model_lasso_data = model_lasso.predict(X_train)
model_lasso_data_test = model_lasso.predict(X_test)
X_train_temp = np.column_stack((X_train,model_lasso_data))
X_test_temp = np.column_stack((X_test,model_lasso_data_test))

model_ridge = Ridge(alpha=9).fit(X_train,y)
model_ridge_data = model_ridge.predict(X_train)
model_ridge_data_test = model_ridge.predict(X_test)
X_train_temp = np.column_stack((X_train_temp,model_ridge_data))
X_test_temp = np.column_stack((X_test_temp,model_ridge_data_test))

def rmse_cv2(model):
    rmse= np.sqrt(-cross_val_score(model, X_train_temp, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


model_ridge = Ridge(alpha=8).fit(X_train_temp,y)
model_lasso = Lasso(alpha=0.0005).fit(X_train_temp,y)

lasso_preds = np.expm1(model_lasso.predict(X_test_temp))
ridge_preds = np.expm1(model_ridge.predict(X_test_temp))


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train_temp, y)
xgb_preds = np.expm1(model_xgb.predict(X_test_temp))

preds = (0.6)*(lasso_preds+ridge_preds)/2 + xgb_preds*(0.4)

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("problem7_attempt2.6_sol.csv", index = False)