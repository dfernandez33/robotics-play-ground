import networkx
from typing import Set, Tuple, List
import math
import heapq


def get_optimal_path(vertices: Set, edges: Set, start: Tuple, goal: Tuple) -> List[Tuple]:
    graph = build_graph(vertices, edges)
    potential_goals = []
    for vertex in vertices:  # find all vertices within 5m of goal
        if distance(vertex, goal) < .05:
            potential_goals.append(vertex)

    path_queue = []
    for potential_goal in potential_goals:  # find shortest path among potential goals
        path = networkx.shortest_path(graph, start, potential_goal, weight='weight')
        cost = networkx.path_weight(graph, path, weight='weight')
        heapq.heappush(path_queue, (cost, path))
    if len(path_queue) == 0:
        return []
    return heapq.heappop(path_queue)[1]


def build_graph(vertices: Set, edges: Set) -> networkx.Graph:
    graph = networkx.Graph()
    graph.add_nodes_from(vertices)
    ebunch = []
    for edge in edges:  # add weight as dictionary for use in networkx
        ebunch.append((edge[0], edge[1], {'weight': edge[2]}))
    graph.add_edges_from(ebunch)
    return graph


def distance(x1: Tuple, x2: Tuple) -> float:
    return math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
