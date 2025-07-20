import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
from collections import deque
import random
import math
from scipy.optimize import minimize

class TRPOBuffer:
    """TRPO经验缓冲区"""
    
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
            'old_log_probs': np.array(self.log_probs)
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

class TRPONetwork(Model):
    """TRPO网络"""
    
    def __init__(self, state_dim, action_dim, action_space='continuous',
                 hidden_dims=[64, 64], activation='tanh'):
        super(TRPONetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        
        # 策略网络
        self.policy_layers = []
        for dim in hidden_dims:
            self.policy_layers.append(layers.Dense(dim, activation=activation))
            
        if action_space == 'continuous':
            self.policy_mean = layers.Dense(action_dim, activation='tanh')
            self.policy_log_std = layers.Dense(action_dim)
        else:
            self.policy_output = layers.Dense(action_dim, activation='softmax')
            
        # 价值网络
        self.value_layers = []
        for dim in hidden_dims:
            self.value_layers.append(layers.Dense(dim, activation=activation))
        self.value_output = layers.Dense(1)
        
    def call(self, states, training=None):
        # 策略网络
        policy_features = states
        for layer in self.policy_layers:
            policy_features = layer(policy_features, training=training)
            
        if self.action_space == 'continuous':
            mean = self.policy_mean(policy_features)
            log_std = self.policy_log_std(policy_features)
            return mean, log_std
        else:
            action_probs = self.policy_output(policy_features)
            return action_probs
            
    def get_value(self, states, training=None):
        # 价值网络
        value_features = states
        for layer in self.value_layers:
            value_features = layer(value_features, training=training)
        value = self.value_output(value_features)
        return value
        
    def get_action_and_log_prob(self, states, training=None):
        """获取动作和对数概率"""
        if self.action_space == 'continuous':
            mean, log_std = self(states, training=training)
            std = tf.exp(log_std)
            
            normal = tfp.distributions.Normal(mean, std)
            actions = normal.sample()
            log_probs = normal.log_prob(actions)
            
            actions = tf.tanh(actions)
            log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - actions**2 + 1e-6), axis=-1, keepdims=True)
            
            return actions, log_probs
        else:
            action_probs = self(states, training=training)
            dist = tfp.distributions.Categorical(probs=action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, log_probs

class TRPOAgent:
    """TRPO智能体"""
    
    def __init__(self, state_dim, action_dim, action_space='continuous',
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, delta=0.01,
                 damping=0.1, max_kl=0.01, buffer_size=2048):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.delta = delta
        self.damping = damping
        self.max_kl = max_kl
        
        # 创建网络
        self.network = TRPONetwork(state_dim, action_dim, action_space)
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # 经验缓冲区
        self.buffer = TRPOBuffer(buffer_size, gamma, gae_lambda)
        
        # 训练统计
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'kl_divergence': [],
            'surrogate_loss': []
        }
        
    def conjugate_gradient(self, states, b, nsteps=10):
        """共轭梯度法求解线性方程"""
        x = tf.zeros_like(b)
        r = b
        p = b
        
        for i in range(nsteps):
            Ap = self.hessian_vector_product(states, p)
            alpha = tf.reduce_sum(r * r) / (tf.reduce_sum(p * Ap) + 1e-8)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = tf.reduce_sum(r_new * r_new) / (tf.reduce_sum(r * r) + 1e-8)
            r = r_new
            p = r + beta * p
            
        return x
        
    def hessian_vector_product(self, states, v):
        """计算Hessian向量积"""
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                if self.action_space == 'continuous':
                    mean, log_std = self.network(states, training=True)
                    std = tf.exp(log_std)
                    normal = tfp.distributions.Normal(mean, std)
                    log_probs = normal.log_prob(mean)
                    log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - tf.tanh(mean)**2 + 1e-6), axis=-1, keepdims=True)
                else:
                    action_probs = self.network(states, training=True)
                    dist = tfp.distributions.Categorical(probs=action_probs)
                    log_probs = dist.log_prob(tf.argmax(action_probs, axis=-1))
                    
                kl = tf.reduce_mean(log_probs)
                
            grads = tape2.gradient(kl, self.network.trainable_variables)
            flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            
        hvp = tape1.gradient(tf.reduce_sum(flat_grads * v), self.network.trainable_variables)
        flat_hvp = tf.concat([tf.reshape(h, [-1]) for h in hvp], axis=0)
        
        return flat_hvp + self.damping * v
        
    @tf.function
    def train_step(self, batch):
        """训练步骤"""
        states = batch['states']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        
        with tf.GradientTape() as tape:
            # 计算当前策略的对数概率
            if self.action_space == 'continuous':
                mean, log_std = self.network(states, training=True)
                std = tf.exp(log_std)
                normal = tfp.distributions.Normal(mean, std)
                log_probs = normal.log_prob(actions)
                log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - tf.tanh(actions)**2 + 1e-6), axis=-1, keepdims=True)
            else:
                action_probs = self.network(states, training=True)
                dist = tfp.distributions.Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                
            # 计算比率
            ratio = tf.exp(log_probs - old_log_probs)
            
            # 代理损失
            surrogate_loss = -tf.reduce_mean(ratio * advantages)
            
            # KL散度
            kl_div = tf.reduce_mean(old_log_probs - log_probs)
            
            # 价值损失
            values = self.network.get_value(states, training=True)
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
        # 计算梯度
        grads = tape.gradient(surrogate_loss, self.network.trainable_variables)
        
        # 展平梯度
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        
        # 使用共轭梯度法求解
        step_dir = self.conjugate_gradient(states, flat_grads)
        
        # 计算步长
        shs = tf.reduce_sum(step_dir * self.hessian_vector_product(states, step_dir))
        lm = tf.sqrt(shs / (2 * self.delta))
        full_step = step_dir / lm
        
        # 应用更新
        self._apply_update(full_step)
        
        return {
            'policy_loss': surrogate_loss,
            'value_loss': value_loss,
            'kl_divergence': kl_div,
            'surrogate_loss': surrogate_loss
        }
        
    def _apply_update(self, flat_update):
        """应用参数更新"""
        start = 0
        for var in self.network.trainable_variables:
            size = tf.reduce_prod(var.shape)
            update = tf.reshape(flat_update[start:start + size], var.shape)
            var.assign_add(update)
            start += size
            
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
            if self.action_space == 'continuous':
                mean, _ = self.network(state, training=False)
                return tf.tanh(mean[0]).numpy(), None, None
            else:
                action_probs = self.network(state, training=False)
                actions = tf.argmax(action_probs, axis=-1)
                return actions[0].numpy(), None, None
                
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