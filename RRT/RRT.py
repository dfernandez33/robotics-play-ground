from typing import Tuple, List
from RRT.display import init_display, draw_connections
from shapely.geometry import LineString
import math
import random


class RRT:

    def __init__(self, start_file: str, goal_file: str, obstacles_file: str, robot_radius: float, arena_width: float,
                 arena_height: float, num_iteration=1000, display=True):
        self.start_file = start_file
        self.goal_file = goal_file
        self.obstacles_file = obstacles_file
        self.num_iterations = num_iteration
        self.robot_radius = robot_radius
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.obstacles = self.__get_obstacles()
        self.start = self.__get_start()
        self.goal = self.__get_goal()
        self.vertices = {self.start}
        self.edges = set()
        self.display = display
        if self.display:
            print('Initializing Display')
            init_display(self.start, self.goal, self.obstacles)

    def build_tree(self) -> Tuple:
        for i in range(self.num_iterations):
            print('Running iteration {}'.format(i))
            x_rand = (random.uniform(self.robot_radius, self.arena_width - self.robot_radius),
                      random.uniform(self.robot_radius, self.arena_height - self.robot_radius))
            while x_rand[0] == 0 or x_rand[0] == 2.921 or x_rand[1] == 0 or x_rand[1] == 2.0066:  # make sure the number is not 0 or 100
                x_rand = (random.uniform(0, 10), random.uniform(0, 10))
            x_nearest, distance = self.__get_nearest_vertex(x_rand)
            if self.__obstacle_free(x_nearest, x_rand):
                self.vertices.add(x_rand)
                self.edges.add((x_nearest, x_rand, distance))
                if self.display:
                    draw_connections((x_nearest, x_rand))

        return self.vertices, self.edges

    def __get_obstacles(self) -> List[Tuple]:
        obstacles = []
        obstacles_file = open(self.obstacles_file)
        for obstacle in obstacles_file.readlines():
            obstacles.append(tuple([float(x) for x in obstacle.split(",")]))

        return obstacles

    def __get_start(self) -> Tuple:
        start_file = open(self.start_file)

        return tuple([float(x) for x in start_file.readline().split(",")])

    def __get_goal(self) -> Tuple:
        goal_file = open(self.goal_file)

        return tuple([float(x) for x in goal_file.readline().split(",")])

    def __obstacle_free(self, start: Tuple, end: Tuple) -> bool:
        for obstacle in self.obstacles:
            x, y, w, h = obstacle
            rect_coord = ((x - self.robot_radius, y - self.robot_radius), (x + w + self.robot_radius, y - self.robot_radius),
                          (x - self.robot_radius, y + h + self.robot_radius), (x + w + self.robot_radius, y + h + self.robot_radius))
            rect_top_left = rect_coord[0]
            rect_bottom_right = rect_coord[3]
            if (rect_top_left[0] <= end[0] <= rect_bottom_right[0] and  # check if point is in obstacle
                    rect_bottom_right[1] <= end[1] <= rect_top_left[1]):
                return False
            new_edge = LineString([end, start])
            if new_edge.intersects(LineString([rect_coord[0], rect_coord[2]])) \
                    or new_edge.intersects(LineString([rect_coord[0], rect_coord[1]])) \
                    or new_edge.intersects(LineString([rect_coord[1], rect_coord[3]])) \
                    or new_edge.intersects(LineString([rect_coord[2], rect_coord[3]])):
                return False
        return True

    def __get_nearest_vertex(self, x_rand: Tuple[float, float]) -> Tuple:
        nearest_dist = math.inf
        nearest_vertex = None
        for v in self.vertices:
            distance = self.__euclidean_dist(x_rand, v)
            if distance < nearest_dist:
                nearest_dist = distance
                nearest_vertex = v
        return nearest_vertex, nearest_dist

    @staticmethod
    def __euclidean_dist(x1: Tuple, x2: Tuple) -> float:
        return math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
