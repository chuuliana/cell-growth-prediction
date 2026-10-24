import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/dataset_cell_growth_4factor.csv")

print(df.head())
print(df.shape)

#show each columns data type and num of non-missing values
df.info()

#counts missing values in each column
print("Missing values per column:")
print(df.isna().sum())

#select the four input columns
X = df[[
    "Temperature [C]",
    "pH",
    "Glucose [mM]",
    "Lactate [mM]"
]]

#select the output we want to predict
y = df["Specific Growth Rate [d^-1]"]

#check their dimensions
print(X.shape)
print(y.shape)

#split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("Training:", X_train.shape, y_train.shape)
print("Testing:", X_test.shape, y_test.shape)

#create pipeline that standardizes inputs then applies linear regression
model_linear = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

#learn from the training conditions and their known growth rates
model_linear.fit(X_train, y_train)

#predict growth rates for the training conditions
y_train_pred = model_linear.predict(X_train)

#first five predictions
print("First five training predictions:")
print(y_train_pred[:5])

#compare known training growth rates with predictions
train_rmse_linear = root_mean_squared_error(y_train, y_train_pred)
train_r2_linear = r2_score(y_train, y_train_pred)

#results
print("Linear training RMSE:", train_rmse_linear)
print("Linear training R2:", train_r2_linear)

##

#evaluate linear pipeline using 5fold cross validation
cv_results_linear = cross_validate(
    model_linear,
    X_train,
    y_train,
    cv=5,
    scoring=("neg_root_mean_squared_error", "r2"),
    return_train_score=True
)

#extract the 5 validation scores for each metric
cv_rmse_linear = -cv_results_linear["test_neg_root_mean_squared_error"]
cv_r2_linear = cv_results_linear["test_r2"]

#average of 5 folds
print("Linear mean CV RMSE:", np.mean(cv_rmse_linear))
print("Linear mean CV R2:", np.mean(cv_r2_linear))

##

#create 2nd degree polynomial regression pipeline
model_poly = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

#evaluate it using same 5fold validation
cv_results_poly = cross_validate(
    model_poly,
    X_train,
    y_train,
    cv=5,
    scoring=("neg_root_mean_squared_error", "r2"),
    return_train_score=True
)

#extract validation scores
cv_rmse_poly = -cv_results_poly["test_neg_root_mean_squared_error"]
cv_r2_poly = cv_results_poly["test_r2"]

#averages of 5 folds
print("Polynomial mean CV RMSE:", np.mean(cv_rmse_poly))
print("Polynomial mean CV R2:", np.mean(cv_r2_poly))

##

#pipeline that standardizes inputs before kNN
model_knn = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=5, weights="uniform")
)

#evaluate pipeline using 5fold cross validation
cv_results_knn = cross_validate(
    model_knn,
    X_train,
    y_train,
    cv=5,
    scoring=("neg_root_mean_squared_error", "r2"),
    return_train_score=True
)

#extract validation scores
cv_rmse_knn = -cv_results_knn["test_neg_root_mean_squared_error"]
cv_r2_knn = cv_results_knn["test_r2"]

#averages
print("kNN mean CV RMSE:", np.mean(cv_rmse_knn))
print("kNN mean CV R2:", np.mean(cv_r2_knn))

##

param_grid_poly = {
    "polynomialfeatures__degree": [1, 2, 3, 4, 5]
}

search_poly = GridSearchCV(
    model_poly,
    param_grid=param_grid_poly,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

search_poly.fit(X_train, y_train)

print("Best polynomial settings:", search_poly.best_params_)
print("Best polynomial CV RMSE:", -search_poly.best_score_)

param_grid_knn = {
    "kneighborsregressor__n_neighbors": list(range(1, 81)),
    "kneighborsregressor__weights": ["uniform", "distance"]
}

search_knn = GridSearchCV(
    model_knn,
    param_grid=param_grid_knn,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

search_knn.fit(X_train, y_train)

print("Best kNN settings:", search_knn.best_params_)
print("Best kNN CV RMSE:", -search_knn.best_score_)

##

results_poly = pd.DataFrame(search_poly.cv_results_)

table_poly = pd.DataFrame({
    "Degree": results_poly["param_polynomialfeatures__degree"],
    "Mean training RMSE": -results_poly["mean_train_score"],
    "Mean CV RMSE": -results_poly["mean_test_score"],
    "CV RMSE standard deviation": results_poly["std_test_score"]
})

print(table_poly.to_string(index=False))

table_poly.to_csv("tables/polynomial_results.csv", index=False)

##

results_knn = pd.DataFrame(search_knn.cv_results_)

table_knn = pd.DataFrame({
    "Neighbors": results_knn["param_kneighborsregressor__n_neighbors"],
    "Weights": results_knn["param_kneighborsregressor__weights"],
    "Mean training RMSE": -results_knn["mean_train_score"],
    "Mean CV RMSE": -results_knn["mean_test_score"],
    "CV RMSE standard deviation": results_knn["std_test_score"]
})

table_knn = table_knn.sort_values(by="Mean CV RMSE")

print(table_knn.head(10).to_string(index=False))

table_knn.to_csv("tables/knn_results.csv", index=False)

##

best_poly = search_poly.best_estimator_
best_knn = search_knn.best_estimator_

y_test_pred_poly = best_poly.predict(X_test)
y_test_pred_knn = best_knn.predict(X_test)

test_rmse_poly = root_mean_squared_error(y_test, y_test_pred_poly)
test_r2_poly = r2_score(y_test, y_test_pred_poly)

test_rmse_knn = root_mean_squared_error(y_test, y_test_pred_knn)
test_r2_knn = r2_score(y_test, y_test_pred_knn)

print("Polynomial test RMSE:", test_rmse_poly)
print("Polynomial test R2:", test_r2_poly)
print("kNN test RMSE:", test_rmse_knn)
print("kNN test R2:", test_r2_knn)

###

fig, ax = plt.subplots(
    figsize=(6.5, 4.0),
    dpi=200,
    layout="constrained"
)

ax.scatter(
    y_test, y_test_pred_poly,
    label="Polynomial: degree 2",
    color="tab:blue",
    alpha=0.7
)

ax.scatter(
    y_test, y_test_pred_knn,
    label="kNN: 10 neighbors, distance",
    color="tab:orange",
    alpha=0.7
)

limits = [y_test.min(), y_test.max()]

ax.plot(
    limits, limits,
    label="Perfect prediction",
    color="black",
    linestyle="--"
)

ax.set_xlabel("Actual specific growth rate [d^-1]", fontsize=10)
ax.set_ylabel("Predicted specific growth rate [d^-1]", fontsize=10)
ax.legend(fontsize=10)

fig.savefig(os.path.join("figures", "test_predictions.png"))
plt.close(fig)

##plots

features = X.columns

fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(10, 7),
    dpi=200,
    layout="constrained"
)

for row in range(2):
    for col in range(2):
        feature = features[2 * row + col]
        ax = axes[row, col]

        ax.scatter(
            X.loc[:, feature], y,
            color="tab:blue",
            alpha=0.7
        )

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Specific growth rate [d^-1]", fontsize=10)

fig.savefig(os.path.join("figures", "growth_relationships.png"))
plt.close(fig)








