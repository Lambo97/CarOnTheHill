import sys
import numpy as np
import pandas as pd
from random import random, randint
from math import floor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from domain import CarOnTheHill
from PIL import Image

import sys
import csv
import time
import os
import datetime

from collections import deque


class ExtraTreesAgent():
    def __init__(self, env):
        self.render = True
        self.discount_factor = .95
        self.buffer = []
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.buffer.append([])
        self.model = None
        self.env = env
        self.action_space = self.env.action_space()

    def memorize(self, state, action, reward, next_state, done):
        # Fill the buffer
        self.buffer[0].append(state)
        self.buffer[1].append(action)
        self.buffer[2].append(reward)
        self.buffer[3].append(next_state)
        self.buffer[4].append(done)

    def train(self):

        X = np.hstack((self.buffer[0], self.buffer[1]))
        y = self.buffer[2]
        print("Training prec...")
        self.model = ExtraTreesRegressor().fit(X, y)

        self.model
        dQ = []
        score = []
        dump(self.model, 'extra_trees/models_Image/Q0.pkl')
        for iteration in range(60):
            y = []

            Q_prev = np.zeros((len(self.buffer[0]), len(self.action_space)))

            for i, a in enumerate(self.action_space):
                testing = np.c_[
                    self.buffer[3], np.repeat(
                        a, len(
                            self.buffer[3]))]
                Q_prev[:, i] = self.model.predict(testing)
            for k, done in enumerate(self.buffer[4]):
                if done:
                    y.append(self.buffer[2][k])
                else:
                    y.append(self.buffer[2][k] +
                             self.discount_factor *
                             np.max(Q_prev[k, :]))

            print("Training {}...".format(iteration))
            self.model.fit(X, y)
            dump(self.model, 'extra_trees/models_Image/Q{}.pkl'.format(iteration + 1))

            old_model = load(
                'extra_trees/models_Image/Q{}.pkl'.format(iteration))
            d = mean_squared_error(self.model.predict(X), old_model.predict(X))
            dQ.append(d)
            score.append(self.test())

        # Save Results
        t = int(time.time())
        np.savetxt(
            "extra_trees/results/dQ_Image_{}.csv".format(t),
            np.array(dQ))
        np.savetxt(
            "extra_trees/results/test_result_Image_{}.csv".format(t),
            np.array(score))

        self.test()
        position_history, speed_history = self.env.get_trajectory()
        np.savetxt(
            "extra_trees/results/position_Image_{}.csv".format(t),
            np.array(position_history))
        np.savetxt(
            "extra_trees/results/speed_Image_{}.csv".format(t),
            np.array(speed_history))

    def get_action(self, state):
        if self.model is None:
            action = self.action_space[randint(0, self.env.action_dim() - 1)]
            return action
        else:
            Q = []
            for a in self.action_space:
                Q.append(self.model.predict([np.hstack((state, [a]))]))
            return self.action_space[np.argmax(Q)]

    def test(self):
        done = False
        state = env.reset()
        stacked_frames, state = stack_states(state, is_new_episode=True)
        state = state.flatten()

        step = 0
        cumulated_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            stacked_frames, next_state = stack_states(
                next_state, stacked_frames)
            next_state = next_state.flatten()
            state = next_state

            step += 1
            cumulated_reward += self.discount_factor ** step * reward
        print(cumulated_reward)
        return cumulated_reward


def preprocess_image(state):
    # Convert to gray-scale and resize it
    image = Image.fromarray(state, 'RGB').convert('L').resize((84, 84))
    # Convert image to array and return it
    array = np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                                image.size[0])
    return array


def stack_states(state, stacked_frames=None, is_new_episode=False):
    # Process the image
    state = preprocess_image(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.uint8)
                                for i in range(4)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(state)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_frames, stacked_state


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    # env = gym.make('Pendulum-v0')
    env = CarOnTheHill(render=True)
    # get size of state and action from environment

    # make A2C agent
    agent = ExtraTreesAgent(env)
    scores, episodes = [], []
    print("Exploring")

    for e in range(1000):
        done = False
        state = env.reset()
        stacked_frames, state = stack_states(state, is_new_episode=True)
        state = state.flatten()
        while not done:

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            stacked_frames, next_state = stack_states(
                next_state, stacked_frames)
            next_state = next_state.flatten()
            agent.memorize(state, [action], reward, next_state, done)

            state = next_state
        print(e)
    print("Training")
    agent.train()
