import numpy as np
import pygame
from math import sqrt, exp, atan2, degrees, pi
from random import random, randint
from time import sleep

# Implementation of the domain


class CarOnTheHill():
    def __init__(self, render=False, state=False):
        self.integaration_step = 0.001
        self.state = state
        self.render = render
        self.surf = self.init_surface()
        self.reset()
        return

    def action_space(self):
        return np.array([-4, 4])

    def sample_action(self):
        x = randint(0, 1)
        return self.action_space()[x]

    def action_dim(self):
        return np.shape(self.action_space())[0]

    def state_dim(self):
        if self.state:
            return 2
        array = self.create_image(self.p, self.s)
        return np.shape(array)

    def __Hill(self, p):
        return p * p + p if p < 0 else p / (sqrt(1 + 5 * p * p))

    def __Hill_p(self, p):
        return 2 * p + 1 if p < 0 else 1 / (1 + 5 * p * p)**(3 / 2)

    def __Hill_pp(self, p):
        return 2 if p < 0 else (-15 * p) / (1 + 5 * p * p)**(5 / 2)

    def __derive_s(self, p, s, action):
        # Compute the derivative of the speed
        m = 1
        g = 9.81
        a = action / (m * (1 + self.__Hill_p(p) ** 2))
        b = (g * self.__Hill_p(p)) / (1 + self.__Hill_p(p) ** 2)
        c = (s * s * self.__Hill_p(p) * self.__Hill_pp(p)) / \
            (1 + self.__Hill_p(p) ** 2)
        return a - b - c

    def __ppoints_to_angle(self, x1, x2):
        dx = x1[1] - x1[0]
        dy = x2[1] - x2[0]
        rads = atan2(-dy, dx)
        rads %= 2 * pi
        degs = degrees(rads)
        return degs

    def __rotate(self, image, rect, angle):
        """Rotate the image while keeping its center."""
        # Rotate the original image without modifying it.
        new_image = pygame.transform.rotate(image, angle)
        # Get a new rect with the center of the old rect.
        rect = new_image.get_rect(center=rect.center)
        return new_image, rect

    def step(self, action):
        # Time step of 0.1s and itegration step 0.001s
        terminal = self.terminal
        for i in range(100):
            # Terminal state
            if abs(self.p) > 1 or abs(self.s) > 3:
                terminal = True
                break
            p_next = self.p + self.integaration_step * self.s
            s_next = self.s + self.integaration_step * \
                self.__derive_s(self.p, self.s, action)
            self.p = p_next
            self.s = s_next

        # Reward
        if self.terminal:
            reward = 0
        # Lose
        elif self.p < -1 or abs(self.s) > 3:
            reward = -1
        # Win
        elif self.p > 1 and abs(self.p) <= 3:
            reward = 1
        # Not in a terminal state
        else:
            reward = 0

        # Avoid infinite episodes
        self.episode_len += 1
        if self.episode_len > 10000:
            terminal = True

        # Add the new state into the trajectory
        self.position_history.append(self.p)
        self.speed_history.append(self.s)

        self.terminal = terminal

        if self.state:
            return (self.p, self.s), reward, self.terminal, ""
        return self.create_image(self.p, self.s), reward, self.terminal, ""

    def reset(self):
        self.p = -0.5
        self.s = 0
        self.terminal = False
        self.episode_len = 0
        self.position_history = []
        self.speed_history = []
        if self.state:
            return (self.p, self.s)
        return self.create_image(self.p, self.s)

    def get_trajectory(self):
        return self.position_history, self.speed_history

    def init_surface(self):
        # Initialization of variables for visualization
        self.canvas_width = 400
        self.canvas_height = 400
        self.screen = pygame.display.set_mode(
            (self.canvas_width, self.canvas_height))
        pt_pos1 = -0.5
        pt_pos2 = 0.5

        # Image loading
        self.car = pygame.image.load("images/car.png")
        pt = pygame.image.load("images/pine_tree.png")
        self.car.convert_alpha()
        pt.convert_alpha()
        size_pt = pt.get_rect().size
        size_car = self.car.get_rect().size
        self.width_car = size_car[0]
        self.height_car = size_car[1]
        width_pt = size_pt[0]
        height_pt = size_pt[1]

        # Initialization of variables related to car on the hill
        step_hill = 2.0 / self.canvas_width

        # Coloring
        color_hill = pygame.Color(0, 0, 0, 0)
        color_shill = pygame.Color(64, 163, 191, 0)
        color_phill = pygame.Color(64, 191, 114, 0)
        color_acc_line = pygame.Color(0, 0, 0, 0)

        # Surface loading
        surf = pygame.Surface((self.canvas_width, self.canvas_height))
        surf.convert()

        # hill function plot
        points = list(np.arange(-1, 1, step_hill))
        hl = list(map(self.__Hill, points))

        # Discretization of the hill function steps
        # Draw the background and the hill function altogether
        range_h = range(self.canvas_height)
        pix = 0
        for h in hl:
            x = pix
            y = ((self.canvas_height) / 2) * (1 + h)

            y = int(round(y))
            for yo in range_h:
                if yo < y:
                    c = color_phill
                elif yo > y:
                    c = color_shill
                surf.set_at((x, self.canvas_height - yo), c)

            surf.set_at((x, self.canvas_height - y), color_hill)
            pix += 1

        # Display pine trees
        surf.blit(pt, (round((self.canvas_width /
                              2) *
                             (1 +
                              pt_pos1)) -
                       width_pt /
                       2, self.canvas_height -
                       round(((self.canvas_height) /
                              2) *
                             (1 +
                              self.__Hill(pt_pos1))) -
                       height_pt))
        surf.blit(pt, (round((self.canvas_width /
                              2) *
                             (1 +
                              pt_pos2)) -
                       width_pt /
                       2, self.canvas_height -
                       round(((self.canvas_height) /
                              2) *
                             (1 +
                              self.__Hill(pt_pos2))) -
                       height_pt))

        return surf

    def create_image(self, position, speed):
        color_acc_line = pygame.Color(0, 0, 0, 0)
        max_speed = 3
        min_speed = -3
        step_hill = 2.0 / self.canvas_width
        max_height_speed = 50
        loc_width_from_bottom = 35
        loc_height_from_bottom = 70
        width_speed = 30
        thickness_speed_line = 3
        screen = self.screen

        # Display the car
        surf = self.surf.copy()
        x_car = round((self.canvas_width / 2) *
                      (1 + position)) - self.width_car / 2
        h_car = self.__Hill(position)
        h_car_next = self.__Hill(position + step_hill)
        y_car = self.canvas_height - \
            round(((self.canvas_height) / 2) * (1 + h_car)) - self.height_car
        angle = self.__ppoints_to_angle(
            (position, position + step_hill), (h_car, h_car_next))
        rot_car, rect = self.__rotate(self.car, pygame.Rect(
            x_car, y_car, self.width_car, self.height_car), 360 - angle)
        surf.blit(rot_car, rect)

        # Display car speed

        # Display black line
        rect = (
            self.canvas_width -
            loc_width_from_bottom -
            width_speed,
            self.canvas_height -
            loc_height_from_bottom,
            width_speed,
            thickness_speed_line)
        surf.fill(color_acc_line, rect)

        pct_speed = abs(speed) / max_speed
        pct_speed = np.clip(pct_speed, 0, 1)
        color_speed = (pct_speed * 255, (1 - pct_speed) * 255, 0)
        height_speed = max_height_speed * (pct_speed)

        loc_width = self.canvas_width - width_speed - loc_width_from_bottom
        loc_height = self.canvas_height - loc_height_from_bottom + \
            thickness_speed_line if speed < 0 else self.canvas_height - loc_height_from_bottom - height_speed
        rect = (loc_width, loc_height, width_speed, height_speed)
        surf.fill(color_speed, rect)

        if self.render:
            screen.blit(surf, (0, 0))
            pygame.display.update()

        return np.array(pygame.surfarray.array3d(surf))
