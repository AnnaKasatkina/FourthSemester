import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

@dataclass
class Node:
    axis: Optional[int] = None  
    median: Optional[float] = None  
    left: Optional["Node"] = None  
    right: Optional["Node"] = None  
    points: Optional[NDArray[np.float64]] = None  
    indices: Optional[NDArray[np.int64]] = None  
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        self.is_leaf = self.points is not None

class KDTree:
    def __init__(self, X, leaf_size=1):
        self.data = np.asarray(X)
        self.leaf_size = leaf_size
        self.root = self._build_tree(np.arange(self.data.shape[0]))

    def _build_tree(self, indices):
        if len(indices) <= self.leaf_size:
            return Node(points=self.data[indices], indices=indices)
        
        variances = np.var(self.data[indices], axis=0)
        axis = np.argmax(variances)
        sorted_indices = indices[np.argsort(self.data[indices, axis])]
        median_index = len(sorted_indices) // 2
        median_value = self.data[sorted_indices[median_index], axis]

        return Node(
            axis=axis,
            median=median_value,
            left=self._build_tree(sorted_indices[:median_index]),
            right=self._build_tree(sorted_indices[median_index:]),
        )
    
    def query(self, X_query, k=1):
        X_query = np.array(X_query)
        neighbors = []

        for point in X_query:
            best = []
            self._search_tree(self.root, point, k, best)
            neighbors.append([idx for _, idx in sorted(best, key=lambda x: -x[0])])

        return neighbors
    
    def _search_tree(self, node, point, k, best):

        if node is None:
            return
        
        if node.is_leaf:
            for index in node.indices:
                dist = np.linalg.norm(point - self.data[index])
                
                if len(best) < k:
                    heapq.heappush(best, (-dist, index))
                elif dist < -best[0][0]:
                    heapq.heappushpop(best, (-dist, index))
            return
        
        axis = node.axis

        if point[axis] <= node.median:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        self._search_tree(first, point, k, best)

        if len(best) < k or abs(point[axis] - node.median) < abs(best[0][0]):
            self._search_tree(second, point, k, best)
            