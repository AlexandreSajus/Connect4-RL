""" Training a DQN Tensorflow agent coded from scratch on the PettingZoo Connect Four environment """


import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pettingzoo.classic import connect_four_v3

from IPython.display import clear_output

os.environ["SDL_VIDEODRIVER"] = "dummy"


class RandomPlayer:
    """
    Player that chooses actions randomly
    """

    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.name = "Random Player"

    def get_action(self, obs_mask, epsilon=None):
        """
        Choose a legal action randomly

        Args:
        - obs_mask: observation and action mask

        Returns:
        - action: action to play
        """
        return self.random_choice_with_mask(np.arange(7), obs_mask["action_mask"])

    def random_choice_with_mask(self, arr, mask):
        """
        Choose a random element from an array, given a mask

        Args:
        - arr: array to choose from
        - mask: mask to apply to the array

        Returns:
        - element: element chosen from the array
        """
        masked_arr = np.ma.masked_array(arr, mask=1 - mask)
        if masked_arr.count() == 0:
            return None
        return self.rng.choice(masked_arr.compressed())


class PlayLeftmostLegal:
    """
    Agent that always plays the leftmost legal action
    """

    def __init__(self):
        self.name = "Left Player"

    def get_action(self, obs_mask, epsilon=None):
        """
        Return the leftmost legal action

        Args:
        - obs_mask: observation and action mask

        Returns:
        - action: action to play
        """
        for i, legal in enumerate(obs_mask["action_mask"]):
            if legal:
                return i
        return None


def play_game(env, agent0, agent1, display=False):
    """
    Plays a game between two agents

    Args:
    - env: environment
    - agent0: first agent
    - agent1: second agent
    - display: whether to display the game

    Returns:
    - Winning agent index (0 or 1) or 0.5 if draw
    """
    done = False
    env.reset()
    obs, _, _, _, _ = env.last()
    while not done:
        for i, agent in enumerate([agent0, agent1]):
            action = agent.get_action(obs, epsilon=0)
            env.step(action)
            if display:
                clear_output(wait=True)
                plt.imshow(env.render())
                plt.show()
            obs, reward, terminated, _, _ = env.last()
            done = terminated
            if np.sum(obs["action_mask"]) == 0:
                if display:
                    print("Draw")
                return 0.5
            if done:
                if display:
                    print(f"Player {i}: {agent.name} won")
                    print(obs["observation"][:, :, 0] - obs["observation"][:, :, 1])
                return i


class EnvAgainstPolicy:
    """
    Simulates a single agent environment against a policy

    Args:
    - env: environment
    - policy: policy to play against
    - first_player: whether the agent is the first player
    """

    def __init__(self, env, policy, first_player=True):
        self.policy = policy
        self.env = env
        self.first_player = first_player
        self.reset()

    def step(self, action):
        """
        Play a single step of the environment

        Args:
        - action: action to play

        Returns:
        - obs: observation
        """
        self.env.step(action)
        obs, reward, terminated, _, _ = self.env.last()
        if terminated:
            self.last_step = obs, reward, True, False, {}
        else:
            action = self.policy.get_action(obs)
            self.env.step(action)
            obs, reward, terminated, _, _ = self.env.last()
            self.last_step = obs, -reward, terminated, False, {}
        return self.last_step

    def reset(self):
        """
        Reset the environment

        Returns:
        - obs: observation
        """

        self.env.reset()
        if not (self.first_player):
            obs, _, _, _, _ = self.env.last()
            action = self.policy.get_action(obs)
            self.env.step(action)

        self.last_step = self.env.last()
        return self.last_step

    def last(self):
        """
        Return the last step of the environment

        Returns:
        - obs: observation
        """
        return self.last_step


def eval_against_policy(env, agent, policy, N_episodes=10, first_player=True):
    """
    Evaluate an agent against a policy

    Args:
    - env: environment
    - agent: agent to evaluate
    - policy: policy to play against
    - N_episodes: number of episodes to play
    - first_player: whether the agent is the first player

    Returns:
    - results: list of rewards
    """
    eval_env = EnvAgainstPolicy(env, policy, first_player=first_player)
    results = []
    for _ in range(N_episodes):
        done = False
        eval_env.reset()
        obs, _, _, _, _ = eval_env.last()
        while not done:
            action = agent.get_action(obs, epsilon=0)
            eval_env.step(action)
            obs, reward, done, _, _ = eval_env.last()
        results.append(reward)
    return results


class DQNAgent:
    """
    DQN agent coded in Tensorflow

    Args:
    - env: environment
    - epsilon: exploration rate
    - gamma: discount factor
    - lr: learning rate
    - batch_size: batch size
    - memory_size: memory size
    - target_update: target update frequency
    - epsilon_decay: epsilon decay rate
    - epsilon_min: minimum epsilon
    - epsilon_max: maximum epsilon
    - seed: random seed
    - tau: soft update parameter
    """

    def __init__(
        self,
        env,
        epsilon=1.0,
        gamma=0.99,
        lr=0.001,
        batch_size=32,
        memory_size=10000,
        target_update=1000,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        epsilon_max=1.0,
        seed=None,
        tau=1e-4,
    ):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.seed = seed
        self.tau = tau

        self.memory = []
        self.memory_counter = 0

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def build_model(self):
        """
        Creates a simple MLP model

        Returns:
        - model: model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(6, 7, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(7, activation="linear"),
            ]
        )
        model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr)
        )
        return model

    def get_action(self, obs, epsilon=None):
        """
        Get an action from the agent

        Args:
        - obs: observation
        - epsilon: exploration rate

        Returns:
        - action: action
        """
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            action = np.random.choice(
                np.arange(7), p=obs["action_mask"] / np.sum(obs["action_mask"])
            )
        else:
            action = np.argmax(
                self.model.predict(obs["observation"][None, ...], verbose=0)
            )
        # If action is not legal, choose a random legal action
        if obs["action_mask"][action] == 0:
            action = np.random.choice(
                np.arange(7), p=obs["action_mask"] / np.sum(obs["action_mask"])
            )
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store a transition in the replay buffer

        Args:
        - obs: observation
        - action: action
        - reward: reward
        - next_obs: next observation
        - done: whether the episode is done
        """
        if self.memory_counter < self.memory_size:
            self.memory.append((obs, action, reward, next_obs, done))
            self.memory_counter += 1
        else:
            self.memory[self.memory_counter % self.memory_size] = (
                obs,
                action,
                reward,
                next_obs,
                done,
            )
            self.memory_counter += 1

    def update_target_model(self):
        """
        Soft update of the target model
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        self.target_model.set_weights(
            [
                self.tau * w + (1 - self.tau) * tw
                for w, tw in zip(weights, target_weights)
            ]
        )

    def train(self):
        """
        Train the agent
        """
        if self.memory_counter < self.batch_size:
            return
        if self.memory_counter % self.target_update == 0:
            self.update_target_model()

        batch = random.sample(self.memory, self.batch_size)

        obs = np.array([b[0]["observation"] for b in batch])
        actions = np.array([b[1] for b in batch])

        rewards = np.array([b[2] for b in batch])

        next_obs = np.array([b[3]["observation"] for b in batch])

        dones = np.array([b[4] for b in batch])

        with tf.GradientTape() as tape:
            q_values = self.model(obs)
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, 7), axis=1)
            next_q_values = self.target_model(next_obs)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            next_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = tf.keras.losses.MSE(q_values, next_q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        Save the model

        Args:
        - path: path to save the model
        """
        self.model.save(path)

    def load(self, path):
        """
        Load the model

        Args:
        - path: path to load the model
        """
        self.model = tf.keras.models.load_model(path, compile=False)
        self.model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr)
        )


# Training
N_episodes = 25000
N_steps = 1000

env = connect_four_v3.env(render_mode="rgb_array")
env = EnvAgainstPolicy(env, RandomPlayer(), first_player=False)

agent = DQNAgent(
    env,
    epsilon=0.8,
    epsilon_decay=0.999999,
    epsilon_min=0.3,
    epsilon_max=1.0,
    seed=42,
    batch_size=1024,
    lr=0.00001,
    tau=1e-4,
)

MODEL_PATH = "models/dqn_agent_1_ep24000_rew0.32_loss0.244.h5"
agent.load(MODEL_PATH)

env.reset()
"""
for episode in range(N_episodes):
    done = False
    obs = env.reset()[0]
    for step in range(N_steps):
        action = agent.get_action(obs)
        if obs["action_mask"][action] == 0:
            break
        env.step(action)
        next_obs, reward, done, _, _ = env.last()
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.train()
        obs = next_obs
        if done:
            break
    if episode % 1000 == 0:
        print(f"Episode {episode}, epsilon {round(agent.epsilon,2)}")
        mean_reward = np.mean(
            eval_against_policy(
                env, agent, RandomPlayer(), N_episodes=100, first_player=True
            )
        )
        print(f"Mean reward: {mean_reward}")
        loss = round(
            agent.model.evaluate(
                np.array([b[0]["observation"] for b in agent.memory]),
                np.array([b[2] for b in agent.memory]),
                verbose=0,
            ),
            3,
        )
        print(f"Loss: {loss}")
        print()
        agent.save(f"models/dqn_agent_1_ep{episode}_rew{mean_reward}_loss{loss}.h5")
"""

# Evaluation
rewards = eval_against_policy(
    env, agent, RandomPlayer(), N_episodes=1000, first_player=True
)
