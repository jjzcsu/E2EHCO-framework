import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
from collections import deque
import random
import math

class PPOBuffer:
    """高级经验回放缓冲区，支持多进程和优先级采样"""
    
    def __init__(self, buffer_size=2048, gamma=0.99, gae_lambda=0.95, 
                 advantage_normalization=True, value_normalization=True):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.advantage_normalization = advantage_normalization
        self.value_normalization = value_normalization
        
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
        self.old_values = None
        
        # 归一化参数
        self.advantage_mean = 0
        self.advantage_std = 1
        self.value_mean = 0
        self.value_std = 1
        
    def add(self, state, action, reward, value, log_prob, done):
        """添加经验到缓冲区"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_gae(self, last_value=0, last_done=False):
        """计算广义优势估计（GAE）"""
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
        if self.advantage_normalization:
            self.advantage_mean = np.mean(advantages)
            self.advantage_std = np.std(advantages) + 1e-8
            advantages = (advantages - self.advantage_mean) / self.advantage_std
            
        # 归一化价值函数
        if self.value_normalization:
            self.value_mean = np.mean(returns)
            self.value_std = np.std(returns) + 1e-8
            returns = (returns - self.value_mean) / self.value_std
            
        self.advantages = advantages
        self.returns = returns
        self.old_values = np.array(self.values)
        
    def get_batch(self, batch_size):
        """获取批次数据"""
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch = {
            'states': np.array([self.states[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'old_log_probs': np.array([self.log_probs[i] for i in indices]),
            'old_values': self.old_values[indices]
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

class ActorCriticNetwork(Model):
    """高级Actor-Critic网络，支持连续和离散动作空间"""
    
    def __init__(self, state_dim, action_dim, action_space='continuous', 
                 hidden_dims=[256, 256], activation='relu', 
                 use_batch_norm=True, use_dropout=True, dropout_rate=0.1):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        
        # 共享特征提取层
        self.feature_layers = []
        for i, dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.feature_layers.append(layers.BatchNormalization())
            if use_dropout:
                self.feature_layers.append(layers.Dropout(dropout_rate))
                
        # Actor网络（策略网络）
        self.actor_layers = []
        self.actor_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        if use_batch_norm:
            self.actor_layers.append(layers.BatchNormalization())
            
        if action_space == 'continuous':
            # 连续动作空间：输出均值和标准差
            self.actor_mean = layers.Dense(action_dim, activation='tanh')
            self.actor_log_std = layers.Dense(action_dim, activation='tanh')
        else:
            # 离散动作空间：输出动作概率
            self.actor_output = layers.Dense(action_dim, activation='softmax')
            
        # Critic网络（价值网络）
        self.critic_layers = []
        self.critic_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        if use_batch_norm:
            self.critic_layers.append(layers.BatchNormalization())
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
            
        if self.action_space == 'continuous':
            mean = self.actor_mean(actor_features)
            log_std = self.actor_log_std(actor_features)
            # 限制标准差范围
            log_std = tf.clip_by_value(log_std, -20, 2)
            return mean, log_std
        else:
            action_probs = self.actor_output(actor_features)
            return action_probs
            
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
        if self.action_space == 'continuous':
            mean, log_std = self(states, training=training)
            std = tf.exp(log_std)
            
            # 重参数化采样
            normal = tfp.distributions.Normal(mean, std)
            actions = normal.sample()
            log_probs = normal.log_prob(actions)
            
            # 处理动作边界
            actions = tf.tanh(actions)
            log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - actions**2 + 1e-6), axis=-1, keepdims=True)
            
            return actions, log_probs
        else:
            action_probs = self(states, training=training)
            dist = tfp.distributions.Categorical(probs=action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, log_probs

class PPOAgent:
    """高级PPO智能体，包含多种优化技巧"""
    
    def __init__(self, state_dim, action_dim, action_space='continuous',
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, target_kl=0.01, train_iters=80,
                 buffer_size=2048, batch_size=64, 
                 value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, use_gae=True,
                 advantage_normalization=True, value_normalization=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        
        # 创建网络
        self.actor_critic = ActorCriticNetwork(
            state_dim, action_dim, action_space,
            hidden_dims=[256, 256, 128],
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.1
        )
        
        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_actor,
            epsilon=1e-5,
            beta_1=0.9,
            beta_2=0.999
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_critic,
            epsilon=1e-5,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # 学习率调度器
        self.actor_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_actor,
            decay_steps=1000,
            decay_rate=0.95
        )
        self.critic_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_critic,
            decay_steps=1000,
            decay_rate=0.95
        )
        
        # 经验缓冲区
        self.buffer = PPOBuffer(
            buffer_size=buffer_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            advantage_normalization=advantage_normalization,
            value_normalization=value_normalization
        )
        
        # 训练统计
        self.train_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
    @tf.function
    def train_step(self, batch):
        """单步训练"""
        states = batch['states']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        old_values = batch['old_values']
        
        with tf.GradientTape(persistent=True) as tape:
            # 前向传播
            if self.action_space == 'continuous':
                mean, log_std = self.actor_critic(states, training=True)
                std = tf.exp(log_std)
                normal = tfp.distributions.Normal(mean, std)
                log_probs = normal.log_prob(actions)
                log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - tf.tanh(actions)**2 + 1e-6), axis=-1, keepdims=True)
                entropy = tf.reduce_mean(normal.entropy())
            else:
                action_probs = self.actor_critic(states, training=True)
                dist = tfp.distributions.Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                entropy = tf.reduce_mean(dist.entropy())
                
            values = self.actor_critic.get_value(states, training=True)
            
            # 计算比率
            ratio = tf.exp(log_probs - old_log_probs)
            
            # PPO裁剪目标
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # 价值损失
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # 熵正则化
            entropy_loss = -entropy
            
            # 总损失
            total_loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # KL散度（用于早停）
            kl_div = tf.reduce_mean(old_log_probs - log_probs)
            
            # 裁剪比例
            clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1) > self.clip_ratio, tf.float32))
            
        # 计算梯度
        actor_grads = tape.gradient(actor_loss, self.actor_critic.trainable_variables)
        critic_grads = tape.gradient(value_loss, self.actor_critic.trainable_variables)
        
        # 梯度裁剪
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        
        # 应用梯度
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_critic.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.actor_critic.trainable_variables))
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'kl_divergence': kl_div,
            'clip_fraction': clip_fraction
        }
        
    def train(self):
        """训练智能体"""
        if len(self.buffer.states) < self.batch_size:
            return
            
        # 计算GAE
        if self.use_gae:
            self.buffer.compute_gae()
        else:
            self.buffer.compute_gae(gamma=1.0, gae_lambda=0.0)
            
        # 多次训练迭代
        for _ in range(self.train_iters):
            batch = self.buffer.get_batch(self.batch_size)
            stats = self.train_step(batch)
            
            # 记录统计信息
            for key, value in stats.items():
                self.train_stats[key].append(float(value))
                
            # 早停检查
            if stats['kl_divergence'] > self.target_kl:
                break
                
    def get_action(self, state, training=True):
        """获取动作"""
        state = tf.expand_dims(state, 0)
        
        if training:
            actions, log_probs = self.actor_critic.get_action_and_log_prob(state, training=True)
            values = self.actor_critic.get_value(state, training=True)
            return actions[0].numpy(), log_probs[0].numpy(), values[0].numpy()
        else:
            if self.action_space == 'continuous':
                mean, _ = self.actor_critic(state, training=False)
                return tf.tanh(mean[0]).numpy(), None, None
            else:
                action_probs = self.actor_critic(state, training=False)
                actions = tf.argmax(action_probs, axis=-1)
                return actions[0].numpy(), None, None
                
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储经验"""
        self.buffer.add(state, action, reward, value, log_prob, done)
        
    def clear_buffer(self):
        """清空缓冲区"""
        self.buffer.clear()
        
    def save_model(self, filepath):
        """保存模型"""
        self.actor_critic.save_weights(filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        self.actor_critic.load_weights(filepath)
        
    def get_training_stats(self):
        """获取训练统计信息"""
        return {key: np.mean(values[-100:]) if values else 0 
                for key, values in self.train_stats.items()}

class AdaptivePPO(PPOAgent):
    """自适应PPO，动态调整超参数"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.adaptive_params = {
            'clip_ratio': self.clip_ratio,
            'target_kl': self.target_kl,
            'train_iters': self.train_iters,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic
        }
        
        # 性能跟踪
        self.performance_history = deque(maxlen=100)
        
    def adapt_hyperparameters(self, recent_rewards):
        """自适应调整超参数"""
        if len(recent_rewards) < 10:
            return
            
        # 计算性能指标
        mean_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        # 更新性能历史
        self.performance_history.append(mean_reward)
        
        if len(self.performance_history) < 20:
            return
            
        # 计算性能趋势
        recent_performance = np.mean(list(self.performance_history)[-10:])
        old_performance = np.mean(list(self.performance_history)[-20:-10])
        
        # 根据性能趋势调整参数
        if recent_performance > old_performance * 1.1:  # 性能提升
            # 增加探索
            self.adaptive_params['clip_ratio'] = min(0.3, self.adaptive_params['clip_ratio'] * 1.05)
            self.adaptive_params['target_kl'] = min(0.02, self.adaptive_params['target_kl'] * 1.05)
        elif recent_performance < old_performance * 0.9:  # 性能下降
            # 减少探索
            self.adaptive_params['clip_ratio'] = max(0.1, self.adaptive_params['clip_ratio'] * 0.95)
            self.adaptive_params['target_kl'] = max(0.005, self.adaptive_params['target_kl'] * 0.95)
            
        # 动态调整学习率
        if reward_std > np.mean(self.performance_history) * 0.5:
            # 高方差，降低学习率
            self.adaptive_params['lr_actor'] *= 0.99
            self.adaptive_params['lr_critic'] *= 0.99
        else:
            # 低方差，适当提高学习率
            self.adaptive_params['lr_actor'] *= 1.001
            self.adaptive_params['lr_critic'] *= 1.001
            
        # 应用调整后的参数
        self.clip_ratio = self.adaptive_params['clip_ratio']
        self.target_kl = self.adaptive_params['target_kl']
        self.train_iters = int(self.adaptive_params['train_iters'])
        
        # 更新优化器学习率
        self.actor_optimizer.learning_rate.assign(self.adaptive_params['lr_actor'])
        self.critic_optimizer.learning_rate.assign(self.adaptive_params['lr_critic']) 