import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from loader import read_excel


def fit(X, y, alpha):
    X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
    X_intercept = X_with_intercept

    # number of columns in matrix of X including intercept
    dimension = X_with_intercept.shape[1]
    # Identity matrix of dimension compatible with our X_intercept Matrix
    A = np.identity(dimension)
    # set first 1 on the diagonal to zero so as not to include a bias term for
    # the intercept
    A[0, 0] = 0
    # We create a bias term corresponding to alpha for each column of X not
    # including the intercept
    A_biased = alpha * A
    thetas = np.linalg.inv(X_with_intercept.T.dot(
        X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
    return thetas


def predict(X, thetas):
    X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = X_predictor.dot(thetas)
    return predictions


data = read_excel()
# data = data.drop(columns=data.columns[0], axis=1)
print(data)
test = data.tail(15)
print("test")
print(test)

train = data.head(135)
print("train")
print(train)

correlation_matrix = np.corrcoef(data)
print("-------------------------------correlation matrix----------------------------------------")
print(correlation_matrix.shape)
print(correlation_matrix)
y = train["N.1"]
print(y)
train = train.drop(["N.1"], axis=1)
print(train)
y_test = test["N.1"].to_numpy()
test = test.drop(["N.1"], axis=1)

test = test.to_numpy()
# test = StandardScaler().fit_transform(test)
# train = StandardScaler().fit_transform(train.to_numpy())
thetas = fit(train, y.to_numpy(), 0.1)
print(thetas)

Y_pred = predict(test, thetas)

err = mean_squared_error(y_test, Y_pred)
print(err)
err_perc = mean_absolute_percentage_error(y_test, Y_pred)
print("err:", str(err_perc * 100) + "%")
plt.plot(np.arange(0, len(y_test)), y_test)
plt.plot(np.arange(0, len(y_test)), Y_pred)

plt.xlabel("index")
plt.ylabel("Number of defects")
plt.show()
