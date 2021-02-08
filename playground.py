from Robomaster.controller import Controller
from RRT.RRT import RRT
from RRT.path import get_optimal_path
from RRT.display import draw_optimal_path
import numpy as np
import pygame
import sys

if __name__ == '__main__':
    robomaster = Controller(connection_mode='direct')
    # while True:
    #     position = input('Type in position command (seperated by space):')
    #     commands = position.split(' ')
    #     print(robomaster.movement.position)
    #     robomaster.movement.move_to_pose(int(commands[0]))
    #     print(robomaster.movement.position)
    rrt = RRT('RRT/start.txt', 'RRT/goal.txt', 'RRT/obstacles.txt', robot_radius=.3302 / 2, arena_width=2.921,
              arena_height=2.0066, num_iteration=2000)
    optimal_path = []
    while len(optimal_path) == 0:
        vertices, edges = rrt.build_tree()
        optimal_path = get_optimal_path(vertices, edges, rrt.start, rrt.goal)
    draw_optimal_path(optimal_path)
    proceed = input('Press enter to continue')
    rotation_matrix = np.eye(3)
    for i, segment in enumerate(optimal_path[1:]):
        world_coordinate = np.append(np.array(segment), [0])  # where we want to go next in the world frame
        wTc = np.append(optimal_path[i], [0])  # this is the current position of the robot with respect to the world frame
        robot_coordinate = np.linalg.inv(rotation_matrix) @ (world_coordinate - wTc)
        robomaster.movement.move_to_pose(distance_x=robot_coordinate[0], distance_y=robot_coordinate[1], speed_xy=.25)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
