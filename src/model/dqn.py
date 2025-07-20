import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import deque
import random
import math

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, state_dim, action_dim, buffer_size=1000000, 
                 alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 存储数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool)
        
        # 优先级
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        
        # 索引管理
        self.ptr = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # 设置优先级
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size):
        """采样经验"""
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            # 基于优先级采样
            priorities = self.priorities[:self.size]
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(self.size, batch_size, p=probs)
            
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'weights': weights,
            'indices': indices
        }
        
        return batch
        
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size

class DuelingDQN(Model):
    """Dueling DQN网络架构"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 activation='relu', use_batch_norm=True, use_dropout=True, dropout_rate=0.1):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.feature_layers = []
        for dim in hidden_dims[:-1]:
            self.feature_layers.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.feature_layers.append(layers.BatchNormalization())
            if use_dropout:
                self.feature_layers.append(layers.Dropout(dropout_rate))
                
        # 价值流（Value Stream）
        self.value_layers = []
        self.value_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        if use_batch_norm:
            self.value_layers.append(layers.BatchNormalization())
        self.value_output = layers.Dense(1)
        
        # 优势流（Advantage Stream）
        self.advantage_layers = []
        self.advantage_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
        if use_batch_norm:
            self.advantage_layers.append(layers.BatchNormalization())
        self.advantage_output = layers.Dense(action_dim)
        
    def call(self, states, training=None):
        # 特征提取
        features = states
        for layer in self.feature_layers:
            features = layer(features, training=training)
            
        # 价值流
        value_features = features
        for layer in self.value_layers:
            value_features = layer(value_features, training=training)
        value = self.value_output(value_features)
        
        # 优势流
        advantage_features = features
        for layer in self.advantage_layers:
            advantage_features = layer(advantage_features, training=training)
        advantage = self.advantage_output(advantage_features)
        
        # 组合价值和优势
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        return q_values

class NoisyDense(layers.Layer):
    """噪声线性层，用于探索"""
    
    def __init__(self, units, sigma_init=0.017, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # 可训练参数
        self.w_mu = self.add_weight(
            name='w_mu',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # 噪声参数
        self.w_sigma = self.add_weight(
            name='w_sigma',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True
        )
        
        # 噪声样本
        self.w_epsilon = None
        self.b_epsilon = None
        
    def call(self, inputs, training=None):
        if training:
            # 生成噪声
            self.w_epsilon = tf.random.normal(self.w_mu.shape)
            self.b_epsilon = tf.random.normal(self.b_mu.shape)
            
            # 应用噪声
            w = self.w_mu + self.w_sigma * self.w_epsilon
            b = self.b_mu + self.b_sigma * self.b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu
            
        output = tf.matmul(inputs, w) + b
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
        
    def reset_noise(self):
        """重置噪声"""
        self.w_epsilon = None
        self.b_epsilon = None

class RainbowDQN(Model):
    """Rainbow DQN网络，结合多种DQN改进"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 activation='relu', use_noisy_layers=True, num_atoms=51, 
                 v_min=-10, v_max=10, use_dueling=True):
        super(RainbowDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_dueling = use_dueling
        self.use_noisy_layers = use_noisy_layers
        
        # 支持范围
        self.support = tf.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 特征提取层
        self.feature_layers = []
        for i, dim in enumerate(hidden_dims[:-1]):
            if use_noisy_layers:
                self.feature_layers.append(NoisyDense(dim, activation=activation))
            else:
                self.feature_layers.append(layers.Dense(dim, activation=activation))
                
        # Dueling架构
        if use_dueling:
            # 价值流
            self.value_layers = []
            if use_noisy_layers:
                self.value_layers.append(NoisyDense(hidden_dims[-1], activation=activation))
                self.value_layers.append(NoisyDense(num_atoms, activation='softmax'))
            else:
                self.value_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
                self.value_layers.append(layers.Dense(num_atoms, activation='softmax'))
                
            # 优势流
            self.advantage_layers = []
            if use_noisy_layers:
                self.advantage_layers.append(NoisyDense(hidden_dims[-1], activation=activation))
                self.advantage_layers.append(NoisyDense(action_dim * num_atoms, activation='softmax'))
            else:
                self.advantage_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
                self.advantage_layers.append(layers.Dense(action_dim * num_atoms, activation='softmax'))
        else:
            # 标准DQN
            self.output_layers = []
            if use_noisy_layers:
                self.output_layers.append(NoisyDense(hidden_dims[-1], activation=activation))
                self.output_layers.append(NoisyDense(action_dim * num_atoms, activation='softmax'))
            else:
                self.output_layers.append(layers.Dense(hidden_dims[-1], activation=activation))
                self.output_layers.append(layers.Dense(action_dim * num_atoms, activation='softmax'))
                
    def call(self, states, training=None):
        # 特征提取
        features = states
        for layer in self.feature_layers:
            features = layer(features, training=training)
            
        if self.use_dueling:
            # Dueling架构
            value_features = features
            for layer in self.value_layers:
                value_features = layer(value_features, training=training)
            value = tf.reshape(value_features, [-1, 1, self.num_atoms])
            
            advantage_features = features
            for layer in self.advantage_layers:
                advantage_features = layer(advantage_features, training=training)
            advantage = tf.reshape(advantage_features, [-1, self.action_dim, self.num_atoms])
            
            # 组合价值和优势
            q_distributions = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        else:
            # 标准DQN
            output = features
            for layer in self.output_layers:
                output = layer(output, training=training)
            q_distributions = tf.reshape(output, [-1, self.action_dim, self.num_atoms])
            
        return q_distributions
        
    def get_q_values(self, states, training=None):
        """获取Q值"""
        q_distributions = self(states, training=training)
        q_values = tf.reduce_sum(q_distributions * self.support, axis=-1)
        return q_values
        
    def reset_noise(self):
        """重置所有噪声层的噪声"""
        for layer in self.feature_layers:
            if isinstance(layer, NoisyDense):
                layer.reset_noise()
                
        if self.use_dueling:
            for layer in self.value_layers + self.advantage_layers:
                if isinstance(layer, NoisyDense):
                    layer.reset_noise()
        else:
            for layer in self.output_layers:
                if isinstance(layer, NoisyDense):
                    layer.reset_noise()

class DQNAgent:
    """高级DQN智能体，包含多种改进技术"""
    
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=1000000, batch_size=32, target_update_freq=1000,
                 use_double_dqn=True, use_dueling_dqn=True, use_prioritized_replay=True,
                 use_noisy_layers=False, use_rainbow=False, num_atoms=51,
                 v_min=-10, v_max=10, hidden_dims=[256, 256]):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling_dqn = use_dueling_dqn
        self.use_prioritized_replay = use_prioritized_replay
        self.use_noisy_layers = use_noisy_layers
        self.use_rainbow = use_rainbow
        
        # 创建网络
        if use_rainbow:
            self.q_network = RainbowDQN(
                state_dim, action_dim, hidden_dims,
                use_noisy_layers=use_noisy_layers,
                num_atoms=num_atoms, v_min=v_min, v_max=v_max,
                use_dueling=use_dueling_dqn
            )
            self.target_network = RainbowDQN(
                state_dim, action_dim, hidden_dims,
                use_noisy_layers=use_noisy_layers,
                num_atoms=num_atoms, v_min=v_min, v_max=v_max,
                use_dueling=use_dueling_dqn
            )
        elif use_dueling_dqn:
            self.q_network = DuelingDQN(state_dim, action_dim, hidden_dims)
            self.target_network = DuelingDQN(state_dim, action_dim, hidden_dims)
        else:
            self.q_network = tf.keras.Sequential([
                layers.Dense(hidden_dims[0], activation='relu', input_shape=(state_dim,)),
                layers.Dense(hidden_dims[1], activation='relu'),
                layers.Dense(action_dim)
            ])
            self.target_network = tf.keras.Sequential([
                layers.Dense(hidden_dims[0], activation='relu', input_shape=(state_dim,)),
                layers.Dense(hidden_dims[1], activation='relu'),
                layers.Dense(action_dim)
            ])
            
        # 复制权重到目标网络
        self.target_network.set_weights(self.q_network.get_weights())
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # 经验缓冲区
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                state_dim, action_dim, buffer_size
            )
        else:
            self.replay_buffer = deque(maxlen=buffer_size)
            
        # 训练统计
        self.train_stats = {
            'loss': [],
            'q_values': [],
            'epsilon': [],
            'td_error': []
        }
        
        # 训练步数
        self.train_step_count = 0
        
    @tf.function
    def train_step(self, batch):
        """单步训练"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        weights = batch.get('weights', tf.ones(self.batch_size))
        
        with tf.GradientTape() as tape:
            if self.use_rainbow:
                # Rainbow DQN训练
                current_q_distributions = self.q_network(states, training=True)
                current_q_distributions = tf.gather_nd(
                    current_q_distributions,
                    tf.stack([tf.range(self.batch_size), actions], axis=1)
                )
                
                # 目标分布
                next_q_distributions = self.target_network(next_states, training=False)
                next_actions = tf.argmax(
                    tf.reduce_sum(next_q_distributions * self.q_network.support, axis=-1),
                    axis=1
                )
                next_q_distributions = tf.gather_nd(
                    next_q_distributions,
                    tf.stack([tf.range(self.batch_size), next_actions], axis=1)
                )
                
                # 投影目标分布
                target_support = rewards[:, None] + self.gamma * self.q_network.support * (1 - dones[:, None])
                target_support = tf.clip_by_value(target_support, self.q_network.v_min, self.q_network.v_max)
                
                # 计算投影
                b = (target_support - self.q_network.v_min) / self.q_network.delta_z
                l = tf.floor(b)
                u = tf.ceil(b)
                
                target_distributions = tf.zeros_like(next_q_distributions)
                
                for i in range(self.q_network.num_atoms):
                    l_mask = tf.cast(tf.equal(l, i), tf.float32)
                    u_mask = tf.cast(tf.equal(u, i), tf.float32)
                    
                    target_distributions += l_mask * next_q_distributions * (u - b)
                    target_distributions += u_mask * next_q_distributions * (b - l)
                    
                # 计算损失
                loss = tf.reduce_mean(weights * tf.keras.losses.categorical_crossentropy(
                    target_distributions, current_q_distributions
                ))
                
            else:
                # 标准DQN训练
                current_q_values = self.q_network(states, training=True)
                current_q = tf.reduce_sum(
                    current_q_values * tf.one_hot(actions, self.action_dim),
                    axis=1
                )
                
                if self.use_double_dqn:
                    # Double DQN
                    next_q_values = self.q_network(next_states, training=False)
                    next_actions = tf.argmax(next_q_values, axis=1)
                    next_q_values_target = self.target_network(next_states, training=False)
                    next_q = tf.reduce_sum(
                        next_q_values_target * tf.one_hot(next_actions, self.action_dim),
                        axis=1
                    )
                else:
                    # 标准DQN
                    next_q_values = self.target_network(next_states, training=False)
                    next_q = tf.reduce_max(next_q_values, axis=1)
                    
                target_q = rewards + self.gamma * next_q * (1 - dones)
                
                # 计算损失
                td_error = target_q - current_q
                loss = tf.reduce_mean(weights * tf.square(td_error))
                
        # 计算梯度
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        
        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        return {
            'loss': loss,
            'q_values': tf.reduce_mean(current_q_values) if not self.use_rainbow else 0,
            'td_error': tf.reduce_mean(tf.abs(td_error)) if not self.use_rainbow else 0
        }
        
    def train(self):
        """训练智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 采样批次
        if self.use_prioritized_replay:
            batch = self.replay_buffer.sample(self.batch_size)
        else:
            batch_data = random.sample(self.replay_buffer, self.batch_size)
            batch = {
                'states': np.array([d[0] for d in batch_data]),
                'actions': np.array([d[1] for d in batch_data]),
                'rewards': np.array([d[2] for d in batch_data]),
                'next_states': np.array([d[3] for d in batch_data]),
                'dones': np.array([d[4] for d in batch_data])
            }
            
        # 训练
        stats = self.train_step(batch)
        
        # 更新优先级
        if self.use_prioritized_replay:
            td_errors = stats['td_error'].numpy()
            self.replay_buffer.update_priorities(batch['indices'], td_errors)
            
        # 更新目标网络
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
            
        # 重置噪声（如果使用噪声层）
        if self.use_noisy_layers:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
            
        # 记录统计信息
        for key, value in stats.items():
            self.train_stats[key].append(float(value))
        self.train_stats['epsilon'].append(self.epsilon)
        
    def get_action(self, state, training=True):
        """获取动作"""
        state = tf.expand_dims(state, 0)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            if self.use_rainbow:
                q_distributions = self.q_network(state, training=False)
                q_values = tf.reduce_sum(q_distributions * self.q_network.support, axis=-1)
            else:
                q_values = self.q_network(state, training=False)
            return tf.argmax(q_values[0]).numpy()
            
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        if self.use_prioritized_replay:
            self.replay_buffer.add(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))
            
    def update_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath):
        """保存模型"""
        self.q_network.save_weights(filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
        
    def get_training_stats(self):
        """获取训练统计信息"""
        return {key: np.mean(values[-100:]) if values else 0 
                for key, values in self.train_stats.items()}

class AdaptiveDQN(DQNAgent):
    """自适应DQN，动态调整超参数"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.adaptive_params = {
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma
        }
        
        # 性能跟踪
        self.performance_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
    def adapt_hyperparameters(self, recent_rewards, recent_losses):
        """自适应调整超参数"""
        if len(recent_rewards) < 20:
            return
            
        # 计算性能指标
        mean_reward = np.mean(recent_rewards)
        mean_loss = np.mean(recent_losses)
        
        # 更新历史
        self.performance_history.append(mean_reward)
        self.loss_history.append(mean_loss)
        
        if len(self.performance_history) < 50:
            return
            
        # 计算趋势
        recent_performance = np.mean(list(self.performance_history)[-20:])
        old_performance = np.mean(list(self.performance_history)[-50:-20])
        
        # 根据性能趋势调整参数
        if recent_performance > old_performance * 1.05:  # 性能提升
            # 减少探索
            self.adaptive_params['epsilon'] = max(self.epsilon_min, self.adaptive_params['epsilon'] * 0.99)
        elif recent_performance < old_performance * 0.95:  # 性能下降
            # 增加探索
            self.adaptive_params['epsilon'] = min(1.0, self.adaptive_params['epsilon'] * 1.01)
            
        # 根据损失调整学习率
        if mean_loss > np.mean(self.loss_history) * 1.2:  # 损失过高
            self.adaptive_params['lr'] *= 0.95  # 降低学习率
        elif mean_loss < np.mean(self.loss_history) * 0.8:  # 损失过低
            self.adaptive_params['lr'] *= 1.05  # 提高学习率
            
        # 应用调整后的参数
        self.epsilon = self.adaptive_params['epsilon']
        self.lr = self.adaptive_params['lr']
        self.gamma = self.adaptive_params['gamma']
        
        # 更新优化器学习率
        self.optimizer.learning_rate.assign(self.lr) 