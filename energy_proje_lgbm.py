import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("1.Grup (Cevahir-Ömer-Osman)/energy_efficiency_data.csv")

################################################
# 1. Exploratory Data Analysis
################################################
df.describe().T
df.nunique()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "float64"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')


    return cat_cols, cat_but_car, num_cols, num_but_cat

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
   num_summary(df, col, plot=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, ["Heating_Load"], col)

for col in num_cols:
    target_summary_with_num(df, ["Cooling_Load"], col)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='b', cmap='RdBu', fmt='.1f')
    plt.show(block=True)



correlation_matrix(df, df.columns)


##################################################
# FE
##################################################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))


X_scaled = StandardScaler().fit_transform(df[df.columns])
df[df.columns] = pd.DataFrame(X_scaled, columns=df.columns)

df.head()


############################################
# model
############################################

X=df.drop(['Heating_Load','Cooling_Load'],axis=1)
y1= df[['Heating_Load']]
y2= df[['Cooling_Load']]
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.33, random_state = 20)


models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y1.values.ravel(), cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
# RMSE: 0.314 (LR)
# RMSE: 0.2533 (KNN)
# RMSE: 0.126 (CART)
# RMSE: 0.1219 (RF)
# RMSE: 0.2799 (SVR)
# RMSE: 0.1201 (GBM)
# RMSE: 0.1139 (XGBoost)
# RMSE: 0.1163 (LightGBM)
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y2.values.ravel(), cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
# RMSE: 0.3461 (LR)
# RMSE: 0.2618 (KNN)
# RMSE: 0.2316 (CART)
# RMSE: 0.1929 (RF)
# RMSE: 0.3065 (SVR)
# RMSE: 0.185 (GBM)
# RMSE: 0.1316 (XGBoost)
# RMSE: 0.144 (LightGBM)


#Hiperparametre Optimizasyonu

lgbm_model = LGBMRegressor(random_state=46)

rmse_y1 = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y1.values.ravel(), cv=5, scoring="neg_mean_squared_error")))
rmse_y2 = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y2.values.ravel(), cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]}

lgbm_gs_best_y1 = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y1_train)

lgbm_gs_best_y2 = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y2_train)

final_model_y1 = lgbm_model.set_params(**lgbm_gs_best_y1.best_params_).fit(X, y1.values.ravel())
final_model_y2 = lgbm_model.set_params(**lgbm_gs_best_y2.best_params_).fit(X, y2.values.ravel())

rmse_y1 = np.mean(np.sqrt(-cross_val_score(final_model_y1, X, y1.values.ravel(), cv=5, scoring="neg_mean_squared_error")))
rmse_y2 = np.mean(np.sqrt(-cross_val_score(final_model_y2, X, y2.values.ravel(), cv=5, scoring="neg_mean_squared_error")))


lgbm_tuned_y1 = LGBMRegressor(**lgbm_gs_best_y1.best_params_).fit(X_train, y1_train.values.ravel())
lgbm_tuned_y2 = LGBMRegressor(**lgbm_gs_best_y2.best_params_).fit(X_train, y2_train.values.ravel())

y1_pred = lgbm_tuned_y1.predict(X_test)
y2_pred = lgbm_tuned_y2.predict(X_test)

np.sqrt(mean_squared_error(y1_test, y1_pred))
np.sqrt(mean_squared_error(y2_test, y2_pred))
#RMSE_y1: 0.04733100035288178
#RMSE_y2: 0.10293092167242815

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(X="Value", y1="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(models, X)
