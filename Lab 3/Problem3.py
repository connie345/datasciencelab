import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smapi
import statsmodels.graphics as smgraph

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew
from scipy.stats import pearsonr

from datetime import datetime

from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from xgboost import XGBRegressor

from lazypredict import Supervised
from lazypredict.Supervised import LazyRegressor




# Helper Functions

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Import the data
test = pd.read_csv("HousePriceData/test.csv",delimiter=',')
train = pd.read_csv("HousePriceData/train.csv", delimiter=',')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

# Log transform the sale price 
train["SalePrice"] = np.log1p(train["SalePrice"])


# Create Plot of unuseful features
# df_categorical=['Street','Condition2','RoofMatl','Heating']
# plt.figure(figsize = (2,2))
# for i in enumerate(df_categorical):
#     plt.subplot(2,2, i[0] + 1);
#     ax = sns.countplot(x=all_data[i[1]])
#     plt.xlabel(i[1]);
# plt.show();

# Drop Columns with low variance
all_data = all_data.drop(['Street','Condition2','RoofMatl','Heating'],axis=1)

# Convert some numerical value into catergorical type
all_data['MSSubClass'] = all_data['MSSubClass'].astype('object')
all_data['MoSold'] = all_data['MoSold'].astype('object')

# Set Ordingal values to ordinal categorical types

bin_map  = {np.nan:0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
bin_map_2 = {np.nan: 0,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
bin_map_3 = {np.nan:0,'No':1,'Mn':2,'Av':3,'Gd':4}
bin_map_4 = {np.nan:0,'MnWw':1,'MnPrv':2,'GdWo':3,'GdPrv':4}
bin_map_5 = {'Fin': 3, 'RFn': 2, 'Unf': 1, np.nan: 0}
bin_map_6 = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}

all_data['ExterQual'] = all_data['ExterQual'].map(bin_map).astype('object')
all_data['ExterCond'] = all_data['ExterCond'].map(bin_map).astype('object')
all_data['BsmtCond'] = all_data['BsmtCond'].map(bin_map).astype('object')
all_data['BsmtQual'] = all_data['BsmtQual'].map(bin_map).astype('object')
all_data['HeatingQC'] = all_data['HeatingQC'].map(bin_map).astype('object')
all_data['KitchenQual'] = all_data['KitchenQual'].map(bin_map).astype('object')
all_data['FireplaceQu'] = all_data['FireplaceQu'].map(bin_map).astype('object')
all_data['GarageCond'] = all_data['GarageCond'].map(bin_map).astype('object')
all_data['GarageQual'] = all_data['GarageQual'].map(bin_map).astype('object')
all_data['PoolQC'] = all_data['PoolQC'].map(bin_map).astype('object')

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(bin_map_2).astype('object')
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(bin_map_2).astype('object')

all_data['BsmtExposure'] = all_data['BsmtExposure'].map(bin_map_3).astype('object')

all_data['Fence'] = all_data['Fence'].map(bin_map_4).astype('object')

all_data['GarageFinish'] = all_data['GarageFinish'].map(bin_map_5).astype('object')

all_data['Functional'] = all_data['Functional'].map(bin_map_6).astype('object')

# Several houses lack GarageYrBlt, we can just replace this with house year built
all_data['GarageYrBlt'].fillna(all_data['YearBuilt'], inplace=True)

# Fill numerical column NaN with mean data and categorical column nans with the mode of the data
col_cat_fillNA = ['Alley', 'GarageType', 'MiscFeature', 'MasVnrType']
for col in col_cat_fillNA:
    all_data[col].fillna('None', inplace=True)
col_num_fillNA = ['MiscFeature', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath',
                    'Functional', 'GarageArea', 'GarageCars', 'TotalBsmtSF',
                    'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']
for col in col_num_fillNA:
    all_data[col].fillna(0, inplace=True)

# Fill LotFrontage with median value of the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Fill MSZoning with mode value of MSSubClass
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# Fill other categorical features with mode value of the neighborhood and OverallQual
col_cat_fillNA = ['Electrical', 'Utilities', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual']
for col in col_cat_fillNA:
    all_data[col] = all_data.groupby(['Neighborhood', 'OverallQual'])[col].transform(lambda x: x.fillna(x.mode()[0]))

# Get dummies variables for remain data
all_data = pd.get_dummies(all_data)

# Add some features
all_data['TotalSF1'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['HasWoodDeck'] = (all_data['WoodDeckSF'] == 0)
all_data['HasOpenPorch'] = (all_data['OpenPorchSF'] == 0) 
all_data['HasEnclosedPorch'] = (all_data['EnclosedPorch'] == 0)
all_data['Has3SsnPorch'] = (all_data['3SsnPorch'] == 0)
all_data['HasScreenPorch'] = (all_data['ScreenPorch'] == 0)
all_data['TotalBathrooms'] = all_data['FullBath'] + all_data['HalfBath'] * 0.5 + all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] * 0.5
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']

# Correct some features
all_data.loc[all_data['GarageYrBlt'] == 2207, 'GarageYrBlt'] = 2007

# Convert all features related to year to age
for i in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']:
    all_data[i] = all_data[i] - all_data[i].min()


# Checking Distributions for outliers
numeric = list(all_data.select_dtypes(include=['float64','int64']).columns)




# Log transform skewed numeric values
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# bad_points = []
# for col in numeric:
#     regression = smapi.ols("data ~ x", data=dict(data=y, x=X_train[col])).fit()
#     # print(regression.summary())
#     testing = regression.outlier_test()
#     # print('Bad data points (bonf(p) < 0.05):')
#     bad_points.extend(testing[testing[testing.columns[2]] < 0.04].index.tolist()) 
#     # # Figure #
#     # figure = smgraph.regressionplots.plot_fit(regression, 1)
#     # # Add line #
#     # line = smgraph.regressionplots.abline_plot(model_results=regression, ax=figure.axes[0])
    
# bad_points = list(set(bad_points))
# X_train = X_train.drop(bad_points)
# y = y.drop(bad_points)

# Removing anomalies
loc = X_train.index[X_train['TotalSF1'] < np.log1p(7500)].tolist()
X_train = X_train.iloc[loc]
y = y.iloc[loc]

rem = X_train.index[(X_train['OverallQual'] == 4) & ((y > 12.4) | (y < 10.5))].tolist()
X_train = X_train.drop(rem)
y = y.drop(rem)

all_data = pd.concat([X_train, X_test], axis=0)



# plt.scatter(X_train['TotalSF1'],y)
# plt.show()

# sns.violinplot(x=train['OverallQual'], y=y)
# plt.show()


# # Feature Selection


# feature_sel_model = SelectFromModel(Lasso(alpha=0.0003,random_state=0))

# feature_sel_model.fit(X_train, y)
# selected_feat = X_train.columns[(feature_sel_model.get_support())]
# print('total features: {}'.format((X_train.shape[1])))
# print('selected features: {}'.format(len(selected_feat)))
# print('features with coefficients shrank to zero: {}'.format(np.sum(feature_sel_model.estimator_.coef_ == 0)))
# X_train = X_train[selected_feat]
# X_test = X_test[selected_feat]


# Create a model

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# Find Best Hypreparameters

# alphas = [0.00001,0.00003,0.00005,0.0001,0.0002,0.0003,0.0005, 0.001]

# cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_lasso)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.title("Lasso")
# plt.savefig("Lasso_MSE_Alpha_3.png")
# plt.cla()
# print(cv_lasso)
# print("Best Lasso: %f" % min(cv_lasso))


# alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15]

# cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_ridge)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.title("Ridge")
# plt.savefig("Ridge_MSE_Alpha_3.png")
# plt.show()
# print(cv_ridge)
# print("Best Ridge: %f" % min(cv_ridge))



# model_lasso = Lasso(alpha=0.0003).fit(X_train,y)
# model_lasso_data_test = np.expm1(model_lasso.predict(X_test))


# solution = pd.DataFrame({"id":test.Id, "SalePrice":model_lasso_data_test})
# solution.to_csv("Attempts/Attmept10.csv", index = False)





# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y, test_size=0.25, random_state=1)

# reg = LazyRegressor(verbose=1,
#                         ignore_warnings=True,
#                         custom_metric=None,
#                         predictions=False,
#                         random_state=1,
#                         # cv=5,
#                         regressors='all')

# models, predictions = reg.fit(X_train2, X_test2, y_train2, y_test2)
# models.sort_values(by='RMSE', ascending=True, inplace=True)
# print(models[:10])


model_lasso = Lasso(alpha=0.0003).fit(X_train,y)
model_lasso_data = model_lasso.predict(X_train)
model_lasso_data_test = model_lasso.predict(X_test)
X_train_temp = np.column_stack((X_train,model_lasso_data))
X_test_temp = np.column_stack((X_test,model_lasso_data_test))

model_ridge = Ridge(alpha=7).fit(X_train,y)
model_ridge_data = model_ridge.predict(X_train)
model_ridge_data_test = model_ridge.predict(X_test)
X_train_temp = np.column_stack((X_train_temp,model_ridge_data))
X_test_temp = np.column_stack((X_test_temp,model_ridge_data_test))

def rmse_cv2(model):
    rmse= np.sqrt(-cross_val_score(model, X_train_temp, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


model_ridge = Ridge(alpha=12).fit(X_train_temp,y)
model_lasso = Lasso(alpha=0.001).fit(X_train_temp,y)

lasso_preds = np.expm1(model_lasso.predict(X_test_temp))
ridge_preds = np.expm1(model_ridge.predict(X_test_temp))

# pred = (0.)(lasso_preds+ridge_preds)/2

# solution = pd.DataFrame({"id":test.Id, "SalePrice":ridge_preds})
# solution.to_csv("Attempts/Attmept17.csv", index = False)

# xgb_params = {
#     'n_estimators': [300,500,1000,2000,3000],
#     'learning_rate': [0.01, 0.1, 0.2,0.3,0.4,0.6,0.8],
#     'max_depth': [2, 3, 4, 5],
#     'min_child_weight': [1, 2, 3],
# }

# # Xgboost grid search
# xgb_grid = GridSearchCV(xgb.XGBRegressor(), xgb_params, cv=10,verbose=2)
# xgb_grid.fit(X_train, y)
# print("XGBoost Best Parameters:", xgb_grid.best_params_)



model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=2, learning_rate=0.1,min_child_weight=3) #the params were tuned using xgb.cv
# print(rmse_cv2(model_xgb))
model_xgb.fit(X_train_temp, y)
xgb_preds = np.expm1(model_xgb.predict(X_test_temp))

# solution = pd.DataFrame({"id":test.Id, "SalePrice":xgb_preds})
# solution.to_csv("Attempts/Attmept16.csv", index = False)

preds = (0.825)*(lasso_preds) + xgb_preds*(0.175)

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("Attempts/Attmept21.csv", index = False)


# dtrain = xgb.DMatrix(X_train, label = y)
# dtest = xgb.DMatrix(X_test)

# params = {"max_depth":2, "eta":0.1, "n_estimators":500,'min_child_weight':3,"learning_rate":0.1}
# model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=100)
# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
# plt.show()


# alphas = [0.00001,0.00003,0.00005,0.0001,0.0002,0.0003,0.0005, 0.001, 0.003, 0.005, 0.01]

# cv_lasso = [rmse_cv2(Lasso(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_lasso)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.title("Lasso")
# plt.savefig("Lasso_MSE_Alpha_Second_3.png")
# plt.cla()
# # print(cv_lasso)
# print("Best Lasso: %f" % min(cv_lasso))


# alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10,11,12,13,14,15,30,45,60]

# cv_ridge = [rmse_cv2(Ridge(alpha = alpha)).mean() for alpha in alphas]
# plt.plot(alphas,cv_ridge)
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.title("Ridge")
# plt.savefig("Ridge_MSE_Alpha_Second_3.png")
# plt.show()
# # print(cv_ridge)
# print("Best Ridge: %f" % min(cv_ridge))

# preds = (0.55)*(lasso_preds+ridge_preds)/2 + xgb_preds*(0.45)

# solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
# solution.to_csv("Attempts/Attmept14.csv", index = False)