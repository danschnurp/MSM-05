import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from loader import read_excel


def add_one(x: np.ndarray) -> np.ndarray:
    """
    It takes an array of numbers and returns an array of numbers, each of which
     is one greater than the corresponding input

    :param x: np.ndarray
    :type x: np.ndarray
    """
    return np.c_[np.ones((x.shape[0], 1)), x]


def create_identity_biased_matrix(shape_size: int, alpha: float) -> np.ndarray:
    """
    > This function creates a square matrix of size `shape_size` with the diagonal elements set to `alpha` and the
    off-diagonal elements set to `1 - alpha`
    for example:
    [[0.  0.  0. ]
     [0.  A   0. ]
     [0.  0.  A  ]]

    :param shape_size: the size of the matrix you want to create
    :type shape_size: int
    :param alpha: the bias parameter
    :type alpha: float
    """
    a = np.identity(shape_size)
    a[0, 0] = 0
    return alpha * a


def fit(samples: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    """
    It takes in a set of samples and their corresponding labels, and returns
     a set of weights that can be used to predict
    the labels of new samples

    :param samples: the training data, a 2D array of shape (n_samples, n_features)
    :type samples: np.ndarray
    :param labels: the labels of the samples
    :type labels: np.ndarray
    :param alpha: the learning rate
    :type alpha: float
    """
    x_with_intercept = add_one(samples)
    identity_biased_matrix = create_identity_biased_matrix(shape_size=x_with_intercept.shape[1], alpha=alpha)
    return np.linalg.inv(x_with_intercept.T.dot(x_with_intercept) +
                         identity_biased_matrix).dot(x_with_intercept.T).dot(labels)


def predict(samples: np.ndarray, thetas_coefficients: np.ndarray) -> np.ndarray:
    """
    > Given a set of samples and a set of thetas, predict the output for each sample

    :param samples: the input data
    :type samples: np.ndarray
    :param thetas_coefficients: the coefficients of the polynomial
    :type thetas_coefficients: np.ndarray
    """
    x_predictor = add_one(samples)
    predictions = x_predictor.dot(thetas_coefficients)
    return predictions


if __name__ == '__main__':
    # change ratio if needed (current is 1:10, test:train)
    train_ratio = 0.9
    # change label column name if needed "Branch" "N.1" "Unnamed: 2"
    label_column_name = "Branch"
    # change read_excel(input_file_path) if needed
    data = read_excel(
        input_file_path="./input_data/firms.xls"
        # input_file_path="./input_data/Wines.xls", sheet_name="DATA"
    )
    # data = data.drop(data.columns[0], axis=1)

    data = data[:].replace(' ', np.nan)
    data = data.dropna()
    data = data.astype('float64')

    test = data.tail(int(np.ceil(len(data) * (1 - train_ratio))))
    print("\n------------------------------- test ----------------------------------------")
    print(test)

    train = data.head(int(np.ceil(len(data) * train_ratio)))
    print("\n-------------------------------  train  ----------------------------------------")
    print(train)

    correlation_matrix = np.corrcoef(data)
    print("\n-------------------------------correlation matrix----------------------------------------")
    print(correlation_matrix.shape)
    print(correlation_matrix)

    train_labels = train[label_column_name]
    train = train.drop([label_column_name], axis=1)

    test_labels = test[label_column_name]
    test = test.drop([label_column_name], axis=1)

    print("\n----test labels----")
    print(test_labels)

    thetas = fit(train, train_labels, 0.1)
    print("\n----thetas----")
    print(thetas)

    Y_pred = predict(test, thetas)

    err = mean_squared_error(test_labels, Y_pred)
    print("\n--- mean_squared_error-----")
    print(err)
    err_perc = mean_absolute_percentage_error(test_labels, Y_pred)
    print("---mean_absolute_percentage_error---")
    print(str(err_perc * 100), "%")
    plt.plot(np.arange(0, len(test_labels)), test_labels, label="original data")
    plt.plot(np.arange(0, len(test_labels)), Y_pred, label="predicted data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
