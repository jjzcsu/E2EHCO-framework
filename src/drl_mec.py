import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import time
from mec import *
from torch.utils.tensorboard import SummaryWriter

# problem = "Pendulum-v0"
# env = gym.make(problem)

user_num = 5
edge_num = 1

env = Env(B=10, USER_NUM=user_num, EDGE_NUM=edge_num, F=5, f=1,
                  Dn=np.random.uniform(300, 500, user_num), Cn=np.random.uniform(900, 1100, user_num),
                  pn=500, pi=100, w1=0.5, w2=0.5)


print("按需使用GPU资源")
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# num_states = env.observation_space.shape[0]
num_states = user_num * 2 + 3
print("Size of State Space ->  {}".format(num_states))
# num_actions = env.action_space.shape[0]
num_actions = user_num * (edge_num + 1) + user_num + edge_num * user_num
print("Size of Action Space ->  {}".format(num_actions))

# upper_bound = env.action_space.high[0]
# lower_bound = env.action_space.low[0]

upper_bound = 1
lower_bound = 0

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on cur        writer.add_scalar("loss", critic_loss, global_step=e)rent one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            # print(critic_loss)
            # writer.add_scalar("loss", tf.reduce_mean(critic_loss), global_step=e)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    # add
    out = layers.Dense(256, activation="relu")(out)

    # outputs = layers.Dense(env.K, activation="sigmoid", kernel_initializer=last_init)(out)
    # outputs = layers.Dense(env.K, activation="tanh", kernel_initializer=last_init)(out)
    # outputs = tf.keras.layers.ReLU()(outputs)

    outputs = layers.Dense(num_actions, activation="sigmoid", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    # outputs = outputs * upper_bound
    outputs = outputs * upper_bound

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.squeeze(legal_action)
    # return [np.squeeze(legal_action)]

def net_process_action(actions):
    action = actions[: user_num * (edge_num+1)]
    action_p = actions[user_num*(edge_num+1): user_num*(edge_num+1) + user_num]
    action_f = actions[user_num*(edge_num+1) + user_num: ]

    '''
    action = action.reshape((user_num, edge_num + 1))
    action_p = action_p * 500
    action_p = np.clip(action_p, 100, 500)
    action_f = action_f.reshape((edge_num, user_num))

    action_index = (action == action.max(axis=1)[:, None]).astype(int)

    for i in range(1, edge_num+1):
        sums = 0
        for j in range(user_num):
            if action_index[j][i] != 0:
                if action_f[i-1][j] == 0:
                    action_f[i-1][j] = 0.01
                sums+= action_f[i-1][j]
        for j in range(user_num):
            if action_index[j][i] != 0:
                action_f[i-1][j] = action_f[i-1][j] / sums * 5
    '''
    action = action.reshape((user_num, edge_num + 1))
    action_p = np.array([500] * user_num)
    action_f = action_f.reshape((edge_num, user_num))

    action_index = (action == action.max(axis=1)[:, None]).astype(int)

    for i in range(1, edge_num + 1):
        cnt = 0
        for j in range(user_num):
            if action_index[j][i] != 0:
                cnt+=1

        for j in range(user_num):
            if action_index[j][i] != 0:
                action_f[i - 1][j] =  5 / cnt

    return action, action_p, action_f

std_dev = 0.05
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001


# critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
# actor_optimizer = tf.keras.optimizers.Adam(actor_lr_schedule)

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# gamma = 0.999
# gamma = 0.2
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

start_time = time.time()

x_min, y_min = get_minimum()
experiment_dir = "tensorboard_data"
writer = SummaryWriter(str(experiment_dir + "/reward_1dev"))

# Takes about 4 min to train
e = 0
for ep in range(total_episodes):

    episodic_reward = 0
    episodic_state = 0
    # while True:

    loc, delay, energy, cost = env.reset(user_num, edge_num)
    prev_state = np.zeros(num_states)
    prev_state[: user_num * 2] = loc.reshape((1, -1))[0]
    prev_state[-3] = delay
    prev_state[-2] = energy
    prev_state[-1] = cost

    the_best = 10000
    for render in range(1000):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state, ou_noise)

            # Recieve state and reward from environment.
            # def step(self, render, action, action_p, action_f):
            # state, reward, done, info = env.step(action)


        action_, action_p, action_f = net_process_action(action)
        loc, delay, energy, cost = env.step(render, action_, action_p, action_f)
        state = np.zeros(num_states)
        state[: user_num*2] = loc.reshape((1, -1))[0]
        state[-3] = delay
        state[-2] = energy
        state[-1] = cost
        reward = 1 / cost
        writer.add_scalar("reward", reward, global_step=e)
        e+=1

        buffer.record((prev_state, action, reward, state))
        episodic_reward -= cost

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state
        the_best = min(the_best, cost)
    episodic_reward /= 1000
    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))


plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()