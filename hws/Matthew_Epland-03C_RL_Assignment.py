
# coding: utf-8

# # TensorFlow Assignment: Reinforcement Learning (RL)

# **[Duke Community Standard](http://integrity.duke.edu/standard.html): By typing your name below, you are certifying that you have adhered to the Duke Community Standard in completing this assignment.**
# 
# Name: Matthew Epland

# ### Short answer
# 
# 1\. One of the fundamental challenges of reinforcement learning is balancing *exploration* versus *exploitation*. What do these two terms mean, and why do they present a challenge?

# Exploration: Randomly trying some new action in order to explore the solution space and find untried, yet possibly better behaviors. Finds new information.
# Exploitation: Take the best action using the current state and model/agent. Exploits currently available information.
# 
# Together exploration and exploitation present a challenge since they are opposite behaviors, but are both required for converging RL. At the start of training the agent knows nothing and has to explore, but by the end when the agent has converged to a max reward (hopefully global, but quite possibly local) it should be exploiting it's learned knowledge. It is up to the agent designer to set the hyperparameters which control this trade-off.

# 2\. Another fundamental reinforcement learning challenge is what is known as the *credit assignment problem*, especially when rewards are sparse. 
# What do we mean by the phrase, and why does this make learning especially difficult?
# How does this interact with reward function design, where we have to be careful that our reward captures the true objective?

# Credit assignment problem: When rewards are sparse, for example when we have to play many additional moves before figuring out if we won or lost the game, it is difficult to propagate the reward value back in time (through space) as it is in some sense spread out over a large time interval (spatial area). This makes RL training hard as any one step will not have a strong reward signaling the best action toward the ultimate goal. You have to be careful when building the reward function in these cases for two reasons. First you have to make sure that there are enough rewards spread out through the states that training is possible, and second you have to make sure that whatever scheme you cooked up to accomplish the first did not somehow subvert the true objective.
# 
# For example if the goal is to move a car to a tile many squares away we can't just put the reward all on the goal tile, as the RL will not converge in a reasonable time. If the reward is spread out, say by radius from the goal tile, it may not point strongly enough in the right direction but instead be about the same for the agent to move in any direction - producing unintended behavior.

# ### Deep SARSA Cart Pole
# 
# [SARSA (state-action-reward-state-action)](https://en.wikipedia.org/wiki/State–action–reward–state–action) is another Q value algorithm that resembles Q-learning quite closely:
# 
# Q-learning update rule:
# \begin{equation}
# Q_\pi (s_t, a_t) \leftarrow (1 - \alpha) \cdot Q_\pi(s_t, a_t) + \alpha \cdot \big(r_t + \gamma \max_a Q_\pi(s_{t+1}, a)\big)
# \end{equation}
# 
# SARSA update rule:
# \begin{equation}
# Q_\pi (s_t, a_t) \leftarrow (1 - \alpha) \cdot Q_\pi(s_t, a_t) + \alpha \cdot \big(r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})\big)
# \end{equation}
# 
# Unlike Q-learning, which is considered an *off-policy* network, SARSA is an *on-policy* algorithm. 
# When Q-learning calculates the estimated future reward, it must "guess" the future, starting with the next action the agent will take. In Q-learning, we assume the agent will take the best possible action: $\max_a Q_\pi(s_{t+1}, a)$. SARSA, on the other hand, uses the action that was actually taken next in the episode we are learning from: $Q_\pi(s_{t+1}, a_{t+1})$. In other words, SARSA learns from the next action he actually took (on policy), as opposed to what the max possible Q value for the next state was (off policy).
# 
# Build an RL agent that uses SARSA to solve the Cart Pole problem. 
# 
# *Hint: You can and should reuse the Q-Learning agent we went over earlier. In fact, if you know what you're doing, it's possible to finish this assignment in about 30 seconds.*

# In[1]:


import random
import gym
import math
import numpy as np
import tensorflow as tf
from collections import deque


# In[2]:


# As hinted, adapted from in class Q-learning tf code
class SARSADNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.state_ = tf.placeholder(tf.float32, shape=[None, 4])
        h = tf.layers.dense(self.state_, units=24, activation=tf.nn.tanh)
        h = tf.layers.dense(h, units=48, activation=tf.nn.tanh)
        self.Q = tf.layers.dense(h, units=2)
        
        self.Q_ = tf.placeholder(tf.float32, shape=[None, 2])
        loss = tf.losses.mean_squared_error(self.Q_, self.Q)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.01, self.global_step, 0.995, 1)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=self.global_step)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.sess.run(self.Q, feed_dict={self.state_: state}))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, next_action, done in minibatch:
            y_target = self.sess.run(self.Q, feed_dict={self.state_: state})
            # Q-learning update rule: ... r_t + \gamma \max_a Q_\pi(s_{t+1}, a)
            # y_target[0][action] = reward if done else reward + self.gamma * np.max(self.sess.run(self.Q, feed_dict={self.state_: next_state})[0])
            
            # SARSA update rule:      ... r_t + \gamma Q_\pi(s_{t+1}, a_{t+1})
            y_target[0][action] = reward if done else reward + self.gamma * self.sess.run(self.Q, feed_dict={self.state_: next_state})[0][next_action]
    
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.sess.run(self.train_step, feed_dict={self.state_: np.array(x_batch), self.Q_: np.array(y_batch)})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                if e % 100 == 0 and not self.quiet:
                    self.env.render()
                if e == 0:
                    action = self.choose_action(state, self.get_epsilon(e)) # first time around have to compute the starting action
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                next_action = self.choose_action(next_state, self.get_epsilon(e+1))
                self.remember(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action # might as well store the next_action here too to speed up the computation
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet:
                    print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = SARSADNCartPoleSolver()
    agent.run()


# Note: you should be able to find that SARSA works much better for the demo we went over during lecture.
# This is not necessarily a general result.
# Q-learning and SARSA tend to do better on different kinds of problems.
