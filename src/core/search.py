from scipy.spatial import cKDTree


class NodeGraph:
    def __init__(self):
        self.V = []
        self.kdtree = None

    def add_node(self, q_rand):
        self.V.append(q_rand)
        self.kdtree = cKDTree(self.V)

    def k_nn_search(self, q_rand, k):
        k_actual = min(k, len(self.V))
        if k_actual == 0:
            return []
        distances, indices = self.kdtree.query(q_rand, k=k_actual)
        if k_actual == 1:
            indices = [indices]
        neighbors = [self.V[i] for i in indices]
        return neighbors

