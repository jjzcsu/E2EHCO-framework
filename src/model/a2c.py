import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
from collections import deque
import random
import math

class A2CBuffer:
    """A2C经验缓冲区"""
    
    def __init__(self, buffer_size=2048, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 存储轨迹数据
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
        self.log_probs = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        
        # 计算后的数据
        self.advantages = None
        self.returns = None
        
    def add(self, state, action, reward, value, log_prob, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_gae(self, last_value=0, last_done=False):
        """计算广义优势估计"""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
                
            delta = self.rewards[t] + self.gamma * next_value * (1 - next_done) - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae_lam
            
        returns = advantages + self.values
        
        # 归一化优势函数
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
        
    def get_batch(self):
        """获取所有数据"""
        batch = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'advantages': self.advantages,
            'returns': self.returns,
            'log_probs': np.array(self.log_probs)
        }
        return batch
        
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class A2CNetwork(Model):
    """A2C网络"""
    
    def __init__(self, state_dim, action_dim, action_space='discrete',
                 hidden_dims=[256, 256], activation='relu'):
        super(A2CNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        
        # 共享特征提取层
        self.feature_layers = []
        for dim in hidden_dims[:-1]:
            self.feature_layers.append(layers.Dense(dim, activation=activation))
            
        # Actor网络
        self.actor_layers = []
        self.actor_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        
        if action_space == 'discrete':
            self.actor_output = layers.Dense(action_dim, activation='softmax')
        else:
            self.actor_mean = layers.Dense(action_dim, activation='tanh')
            self.actor_log_std = layers.Dense(action_dim)
            
        # Critic网络
        self.critic_layers = []
        self.critic_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        self.critic_output = layers.Dense(1)
        
    def call(self, states, training=None):
        # 特征提取
        features = states
        for layer in self.feature_layers:
            features = layer(features, training=training)
            
        # Actor分支
        actor_features = features
        for layer in self.actor_layers:
            actor_features = layer(actor_features, training=training)
            
        if self.action_space == 'discrete':
            action_probs = self.actor_output(actor_features)
            return action_probs
        else:
            mean = self.actor_mean(actor_features)
            log_std = self.actor_log_std(actor_features)
            return mean, log_std
            
    def get_value(self, states, training=None):
        # 特征提取
        features = states
        for layer in self.feature_layers:
            features = layer(features, training=training)
            
        # Critic分支
        critic_features = features
        for layer in self.critic_layers:
            critic_features = layer(critic_features, training=training)
            
        value = self.critic_output(critic_features)
        return value
        
    def get_action_and_log_prob(self, states, training=None):
        """获取动作和对数概率"""
        if self.action_space == 'discrete':
            action_probs = self(states, training=training)
            dist = tfp.distributions.Categorical(probs=action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, log_probs
        else:
            mean, log_std = self(states, training=training)
            std = tf.exp(log_std)
            
            normal = tfp.distributions.Normal(mean, std)
            actions = normal.sample()
            log_probs = normal.log_prob(actions)
            
            actions = tf.tanh(actions)
            log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - actions**2 + 1e-6), axis=-1, keepdims=True)
            
            return actions, log_probs

class A2CAgent:
    """A2C智能体"""
    
    def __init__(self, state_dim, action_dim, action_space='discrete',
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, buffer_size=2048):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.network = A2CNetwork(state_dim, action_dim, action_space)
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # 经验缓冲区
        self.buffer = A2CBuffer(buffer_size, gamma, gae_lambda)
        
        # 训练统计
        self.train_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
        
    @tf.function
    def train_step(self, batch):
        """训练步骤"""
        states = batch['states']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['log_probs']
        
        with tf.GradientTape() as tape:
            # 前向传播
            if self.action_space == 'discrete':
                action_probs = self.network(states, training=True)
                dist = tfp.distributions.Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                entropy = tf.reduce_mean(dist.entropy())
            else:
                mean, log_std = self.network(states, training=True)
                std = tf.exp(log_std)
                normal = tfp.distributions.Normal(mean, std)
                log_probs = normal.log_prob(actions)
                log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - tf.tanh(actions)**2 + 1e-6), axis=-1, keepdims=True)
                entropy = tf.reduce_mean(normal.entropy())
                
            values = self.network.get_value(states, training=True)
            
            # 计算损失
            actor_loss = -tf.reduce_mean(advantages * log_probs)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_loss = -entropy
            
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
        # 计算梯度
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        
        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss
        }
        
    def train(self):
        """训练智能体"""
        if len(self.buffer.states) == 0:
            return
            
        # 计算GAE
        self.buffer.compute_gae()
        
        # 获取批次数据
        batch = self.buffer.get_batch()
        
        # 训练
        stats = self.train_step(batch)
        
        # 记录统计信息
        for key, value in stats.items():
            self.train_stats[key].append(float(value))
            
        # 清空缓冲区
        self.buffer.clear()
        
    def get_action(self, state, training=True):
        """获取动作"""
        state = tf.expand_dims(state, 0)
        
        if training:
            actions, log_probs = self.network.get_action_and_log_prob(state, training=True)
            values = self.network.get_value(state, training=True)
            return actions[0].numpy(), log_probs[0].numpy(), values[0].numpy()
        else:
            if self.action_space == 'discrete':
                action_probs = self.network(state, training=False)
                actions = tf.argmax(action_probs, axis=-1)
                return actions[0].numpy(), None, None
            else:
                mean, _ = self.network(state, training=False)
                return tf.tanh(mean[0]).numpy(), None, None
                
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储经验"""
        self.buffer.add(state, action, reward, value, log_prob, done)
        
    def save_model(self, filepath):
        """保存模型"""
        self.network.save_weights(filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        self.network.load_weights(filepath)
        
    def get_training_stats(self):
        """获取训练统计信息"""
        return {key: np.mean(values[-100:]) if values else 0 
                for key, values in self.train_stats.items()} 