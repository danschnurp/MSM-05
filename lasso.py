import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def read_data_cork(file_path, sheetname):
    df = pd.read_excel(file_path, sheet_name=sheetname)
    X = df.to_numpy()

    # selecting and removing a column with a categorical variable
    categories_column_index = 3
    Y = X[:, categories_column_index]
    X = np.delete(X, categories_column_index, 1)

    # remove unnecessary columns
    column_to_remove = 0
    X = np.delete(X, column_to_remove, 1)
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)

    return X, Y


def split_data(X, Y, pomer):
    x_train = X[0:pomer,:]
    y_train = Y[0:pomer]
    x_test = X[pomer:,:]
    y_test = Y[pomer:]

    return x_train, y_train, x_test, y_test

def add_one(X):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

X, Y = read_data_cork("CorkStoppers.xlsx", "Data")
X = add_one(X)


# Split data into groups
pomer = len(Y)*0.9
pomer = int(pomer)
x_train, y_train, x_test, y_test = split_data(X, Y, pomer)

#Lasso regrese
clf = linear_model.Lasso()
clf.fit(x_train, y_train)

reg = linear_model.Lasso(alpha=1)
reg.fit(x_train, y_train)
print("Lasso regression Betas", reg.coef_)

print('R squared training set', round(reg.score(x_train, y_train)*100, 2))
print('R squared test set', round(reg.score(x_test, y_test)*100, 2))

# Training data
pred_train = reg.predict(x_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred = reg.predict(x_test)
mse_test = mean_squared_error(y_test, pred)
print('MSE test set', round(mse_test, 2))

err = np.sum(abs(reg.predict(x_test)-y_test)/y_test)/len(y_test)
print(err * 100, "%")

plt.plot(np.arange(0,15,1), y_test)
plt.plot(np.arange(0,15,1), reg.predict(x_test))
plt.show()



