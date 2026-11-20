from shapely.geometry import Point, Polygon, box


class Environment2D:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.boundary = box(x_min, y_min, x_max, y_max)
        self.obstacles = []

    def add_obstacle(self, vertices):
        obstacle = Polygon(vertices)
        self.obstacles.append(obstacle)

    def is_state_valid(self, x, y):  # collision checker
        point = Point(x, y)

        if not self.boundary.contains(point):
            return False

        for obs in self.obstacles:
            if obs.contains(point):
                return False

        return True

    def is_edge_valid(self, p1, p2): # check if the line between p1 and p2 is valid (collision free) or not
        from shapely.geometry import LineString
        line = LineString([p1, p2])

        for obs in self.obstacles:
            if line.intersects(obs):
                return False
        return True
