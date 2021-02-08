from typing import Tuple, List, Dict
import pygame

screen_size = (1164, 800)
scale_factor = (1164 / 2.921, 800 / 2.0066)  # since window is size 800, 800 and the environment is 2.921x2.0066
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
purple = (128, 0, 128)
red = (255, 0, 0)
screen = pygame.display.set_mode(screen_size)


def init_display(start: Tuple, goal: Tuple, obstacles: List[Tuple]):
    screen.fill(white)
    scaled_start = (start[0] * scale_factor[0], start[1] * scale_factor[1])
    scaled_goal = (goal[0] * scale_factor[0], goal[1] * scale_factor[1])
    pygame.draw.rect(screen, green, [scaled_start[0], scaled_start[1], .1778 * scale_factor[0],
                                     .1778 * scale_factor[1]], 0)
    pygame.draw.rect(screen, blue, [scaled_goal[0], scaled_goal[1],
                                    .1778 * scale_factor[0], .1778 * scale_factor[1]], 0)
    for obstacle in obstacles:
        pygame.draw.rect(screen, black, (obstacle[0] * scale_factor[0], obstacle[1] * scale_factor[1],
                                         obstacle[2] * scale_factor[0], obstacle[3] * scale_factor[1]), 0)

    pygame.display.update()


def draw_connections(new_edge: Tuple[Tuple[float, float], Tuple[float, float]], color=purple):
    scaled_e = ((new_edge[0][0] * scale_factor[0], new_edge[0][1] * scale_factor[1]),
                (new_edge[1][0] * scale_factor[0], new_edge[1][1] * scale_factor[1]))
    pygame.draw.lines(screen, color, True, list(scaled_e), 5)
    pygame.display.update()


def draw_optimal_path(path: List[Tuple]):
    for i in range(len(path) - 1):
        curr_edge = (path[i], path[i + 1])
        draw_connections(curr_edge, red)
