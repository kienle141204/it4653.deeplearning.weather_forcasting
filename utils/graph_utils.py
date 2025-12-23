import numpy as np
from scipy.spatial.distance import cdist


def build_adjacency_matrix(grid_size, method="distance", threshold=None, k=None):
    H, W = grid_size
    N = H * W

    coords = []
    for i in range(H):
        for j in range(W):
            coords.append([i, j])
    coords = np.array(coords)

    dist_matrix = cdist(coords, coords, metric="euclidean")

    if method == "distance":
        if threshold is None:
            threshold = 1.5
        adj = (dist_matrix <= threshold).astype(float)
        np.fill_diagonal(adj, 0)
    elif method == "knn":
        if k is None:
            k = 8
        adj = np.zeros((N, N))
        for i in range(N):
            indices = np.argsort(dist_matrix[i])[1 : k + 1]
            adj[i, indices] = 1.0
    else:
        raise ValueError("Unknown adjacency method")

    degree = np.sum(adj, axis=1)
    degree = np.where(degree > 0, degree, 1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = degree_inv_sqrt @ adj @ degree_inv_sqrt

    return adj_normalized.astype(np.float32)


