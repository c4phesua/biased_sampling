from core.graph import NodeGraph


class PRM2D:
    def __init__(self, env, graph: NodeGraph, sampler, q_start, q_goal):
        self.env = env
        self.graph = graph
        self.sampler = sampler
        self.q_start = q_start
        self.q_goal = q_goal
        self.graph.add_node(q_start)
        self.graph.add_node(q_goal)

    def run(self, sample_num, k: int = 5):
        while len(self.graph.V) < sample_num + 2:
            q_rand = self.sampler()
            if self.env.is_state_valid(q_rand[0], q_rand[1]):
                self.graph.add_node(q_rand)
                neighbors = self.graph.k_nn_search(q_rand, k=k)
                for q_near in neighbors:
                    if self.env.is_edge_valid(q_rand, q_near):
                        self.graph.add_edge(q_rand, q_near)
        return self.graph.get_shortest_path(self.q_start, self.q_goal)
