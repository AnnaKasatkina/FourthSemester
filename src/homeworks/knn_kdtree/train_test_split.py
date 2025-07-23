import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    test_count = int(len(X) * test_size)
    
    X_train, X_test = X[indices[:-test_count]], X[indices[-test_count:]]
    y_train, y_test = y[indices[:-test_count]], y[indices[-test_count:]]

    return X_train, X_test, y_train, y_test
