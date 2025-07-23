import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std = np.where(self.std == 0, 1, self.std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        self.median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr = q3 - q1

        self.iqr = np.where(self.iqr == 0, 1, self.iqr)

    def transform(self, X):
        X_scaled = (X - self.median) / self.iqr
        
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    