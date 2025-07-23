import numpy as np
from collections import Counter
from kd_tree import KDTree

class KNNClassifier:
    def __init__(self, k, leaf_size=10):
        self.k = k
        self.leaf_size = leaf_size
        self.tree = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.y_train = np.asarray(y_train)
        self.tree = KDTree(X_train, leaf_size=self.leaf_size)

    def predict_proba(self, X_test):
        neighbors = self.tree.query(X_test, k=self.k)
        unique_classes = np.unique(self.y_train)
        proba = np.zeros((len(X_test), len(unique_classes)))

        for i, indices in enumerate(neighbors):
            neighbors_classes = self.y_train[indices]
            class_counts = Counter(neighbors_classes)
            
            for j, cls in enumerate(unique_classes):
                proba[i, j] = class_counts[cls] / self.k

        return proba
    
    def predict(self, X_test):
        proba = self.predict_proba(X_test) 
        class_indices = np.argmax(proba, axis=1)
        unique_classes = np.unique(self.y_train)

        return unique_classes[class_indices]
