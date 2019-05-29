import numpy as np
import argparse
import time
import keras.backend as K

from random import random, randint
from PIL import Image
from collections import deque

from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda
from keras.models import load_model
from keras.optimizers import Adam

from domain import CarOnTheHill
from memory import MemoryBuffer
import tensorflow as tf
import os

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNModel:
    def __init__(self, state_dim, action_dim, lr,):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Initialize Deep Q-Network
        self.model = self.build_network()
        self.model.summary()
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.build_network()
        self.target_model.compile(Adam(lr), 'mse')
        self.update_target()

    def build_network(self):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))

        # Network
        x = Dense(10, activation='relu')(inp)
        x = Dense(10, activation='relu')(x)
        x = Dense(self.action_dim, activation='linear')(x)

        return Model(inp, x)

    def update_target(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        self.target_model.set_weights(self.model.get_weights())
        return

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        return self.model.fit(inp, targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        print(inp)
        exit()
        return self.model.predict(inp)

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(inp)

    def save(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)


class DQNAgent:
    def __init__(
            self,
            env,
            gamma=0.95,
            buffer_size=100000,
            batch_size=32,
            epsilon_decay=2e-6,
            n_episodes=10000,
            lr=3e-4,
            per=False):
        # Set Hyperparameter
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.per = per

        # Create Memory buffer
        self.memory = MemoryBuffer(buffer_size=self.buffer_size, with_per=per)

        # Create Model
        self.model = DQNModel(
            state_dim=(
                2,
            ),
            action_dim=env.action_dim(),
            lr=lr)

    def get_action(self, state, test=False):
        actions = self.env.action_space()
        n_actions = self.env.action_dim()
        if random() < self.epsilon and not test:
            index = randint(0, n_actions - 1)
            return actions[index], index
        else:
            Q_values = self.model.predict(np.array([state]))[0]
            action_index = np.argmax(Q_values)
            return actions[action_index], action_index

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0.1

    def memorize(self, state, action, reward, done, next_state):
        """ Store experience in memory buffer
        """
        if(self.per):
            q_val = self.model.predict(np.array([state]))
            q_val_t = self.model.target_predict(np.array([next_state]))
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.memory.memorize(state, action, reward, done, next_state, td_error)

    def learn(self):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, next_s, idx = self.memory.sample_batch(self.batch_size)
        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.model.predict(s)
        next_q = self.model.predict(next_s)
        q_targ = self.model.target_predict(next_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.per):
                # Update PER Sum Tree
                self.memory.update(idx[i], abs(old_q - q[i, a[i]]))

        # Train on batch

        h = self.model.fit(s, q)
        return h

    def train(self):
        test_score = []
        loss_history = []
        step = 0
        for episode in range(self.n_episodes):
            done = False
            state = self.env.reset()
            loss = []
            while not done:
                action, action_index = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memorize(state, action_index, reward, done, next_state)

                state = next_state
                step += 1

                if self.memory.is_full():
                    history = self.learn()
                    loss.append(history.history['loss'][0])
                    self.update_epsilon()
                    if step % 500 == 0:
                        self.model.update_target()

            # Compute the mean of the loss over the
            # preceeding episode
            if not loss:
                loss.append(0)
            mean_loss = np.mean(loss)
            loss_history.append(mean_loss)

            print(
                "EPISODE {} ENDED | PREC REWARD {} | EPSILON {} | MEAN LOSS {} | step {}".format(
                    episode,
                    reward,
                    self.epsilon,
                    mean_loss,
                    step))
            if self.memory.is_full() and episode % 50 == 0:
                test_score.append(self.test())

        if self.per:
            p = "PER"
        else:
            p = ""

        t = int(time.time())

        np.savetxt("results/test_result_State_{}_{}.csv".format(p, t),
                   np.array(test_score))
        np.savetxt("results/loss_result_State_{}_{}.csv".format(p, t),
                   np.array(loss_history))

        # Save the trajectory
        self.test()
        position_history, speed_history = self.env.get_trajectory()
        np.savetxt("results/position_State_{}_{}.csv".format(p, t),
                   np.array(position_history))
        np.savetxt("results/speed_State_{}_{}.csv".format(p, t),
                   np.array(speed_history))
        self.model.save("model/final_model_State_{}_{}.h5".format(p, t))

    def test(self):
        done = False
        state = self.env.reset()
        step = 0
        test_s = 0
        while not done:
            action, _ = self.get_action(state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            step += 1
            test_s += self.gamma ** step * reward
            state = next_state
        print(test_s, step)
        return test_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--buffer', type=int, default=100000)
    parser.add_argument('--lr', type=int, default=3e-4)
    parser.add_argument(
        '--per',
        help='Prioritized Experience Replay',
        action='store_true')
    args = parser.parse_args()

    print(args)

    agent = DQNAgent(
        CarOnTheHill(
            state=True),
        gamma=args.gamma,
        n_episodes=args.episodes,
        buffer_size=args.buffer,
        lr=args.lr,
        per=args.per)
    # agent.train()
    agent.model.load_weights('model/final_model_State_PER_1559067178.h5')
    agent.test()
