import random
import numpy as np
from sklearn.metrics import pairwise_distances
from hypothesis import given
import hypothesis.strategies as st
from kd_tree import KDTree

def brute_force_knn(X_train, X_test, k=3):
    distances = pairwise_distances(X_test, X_train, metric="euclidean")
    return np.argsort(distances, axis=1)[:, :k]

class TestKDTree:
    @given(
        st.integers(min_value=100, max_value=200),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=15),
    )
    def test_query(self, train_size, dim, leaf_size, k):
        X_train = np.random.rand(train_size, dim) * 100
        X_test = np.random.rand(30, dim) * 100

        tree = KDTree(X_train, leaf_size)
        kd_neighbors = tree.query(X_test, k)
        brute_neighbors = brute_force_knn(X_train, X_test, k)

        for kd, brute in zip(kd_neighbors, brute_neighbors):
            assert set(kd) == set(brute)

        for i, point in enumerate(X_test):
            kd_distances = sorted([np.linalg.norm(point - X_train[idx]) for idx in kd_neighbors[i]])
            brute_distances = sorted([np.linalg.norm(point - X_train[idx]) for idx in brute_neighbors[i]])

            assert np.allclose(kd_distances, brute_distances, atol=1e-5)