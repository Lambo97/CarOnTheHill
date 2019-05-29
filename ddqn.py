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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Just disables the warning for CPU instruction set,
#  doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNModel:
    def __init__(self, state_dim, action_dim, lr):
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

        x = Conv2D(32, 8, strides=(4, 4),
                   padding='valid',
                   activation='relu',
                   input_shape=self.state_dim)(inp)

        x = Conv2D(64, 4, strides=(2, 2),
                   padding='valid',
                   activation='relu')(x)

        x = Conv2D(64, 3, strides=(1, 1),
                   padding='valid',
                   activation='relu')(x)

        # Flatten the convolution output
        x = Flatten()(x)

        # Dense layer
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.action_dim, activation='linear')(x)

        return Model(inp, x)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
        return

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        return self.model.fit(inp, targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
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
            img_size=84,
            buffer_size=100000,
            batch_size=32,
            epsilon_decay=2e-6,
            n_episodes=10000,
            lr=3e-4,
            per=False):
        # Set Hyperparameter
        self.env = env
        self.gamma = gamma
        self.img_size = img_size
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
                img_size,
                img_size,
                4),
            action_dim=env.action_dim(),
            lr=lr)

    def preprocess_image(self, state):
        # Convert to gray-scale and resize it
        image = Image.fromarray(
            state, 'RGB').convert('L').resize(
            (self.img_size, self.img_size))
        # Convert image to array and return it
        array = np.asarray(
            image.getdata(),
            dtype=np.uint8).reshape(
            image.size[1],
            image.size[0])
        return array

    def stack_states(self, state, is_new_episode=False):
        # Process the image
        state = self.preprocess_image(state)

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros(
                (self.img_size, self.img_size), dtype=np.uint8) for i in range(4)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            self.stacked_frames.append(state)
            self.stacked_frames.append(state)
            self.stacked_frames.append(state)
            self.stacked_frames.append(state)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different
            # frames)
            stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

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
        with tf.device('/device:GPU:0'):
            h = self.model.fit(s, q)
        return h

    def train(self):
        test_score = []
        loss_history = []
        frame = 0
        for episode in range(self.n_episodes):
            done = False
            state = self.env.reset()
            state = self.stack_states(state, is_new_episode=True)
            loss = []
            while not done:
                action, action_index = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.stack_states(next_state)
                self.memorize(state, action_index, reward, done, next_state)
                state = next_state
                frame += 1
                if self.memory.is_full():
                    history = self.learn()
                    loss.append(history.history['loss'][0])
                    self.update_epsilon()
                    if frame % 1000 == 0:
                        self.model.update_target()

            # Compute the mean of the loss over the
            # preceeding episode
            if not loss:
                loss.append(0)
            mean_loss = np.mean(loss)
            loss_history.append(mean_loss)

            print(
                "EPISODE {} ENDED | PREC REWARD {} | EPSILON {} | MEAN LOSS {} | FRAME {}".format(
                    episode,
                    reward,
                    self.epsilon,
                    mean_loss,
                    frame))
            if self.memory.is_full() and episode % 50 == 0:
                test_score.append(self.test())

        if self.per:
            p = "_PER_"
        else:
            p = ""

        t = int(time.time())

        np.savetxt("results/test_result{}_{}.csv".format(p, t),
                   np.array(test_score))
        np.savetxt("results/loss_result{}_{}.csv".format(p, t),
                   np.array(loss_history))

        # Save the trajectory
        self.test()
        position_history, speed_history = self.env.get_trajectory()
        np.savetxt("results/position_{}_{}.csv".format(p, t),
                   np.array(position_history))
        np.savetxt("results/speed_{}_{}.csv".format(p, t),
                   np.array(speed_history))

        self.model.save("model/final_model{}_{}.h5".format(p, t))

    def test(self):
        done = False
        state = self.env.reset()
        state = self.stack_states(state, is_new_episode=True)
        step = 0
        test_s = 0
        while not done:
            action, _ = self.get_action(state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.stack_states(next_state)
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
            render=True),
        gamma=args.gamma,
        n_episodes=args.episodes,
        buffer_size=args.buffer,
        lr=args.lr,
        per=args.per)
    # agent.train()
    agent.model.load_weights('model/final_model_PER__1558916444.h5')
    agent.test()
