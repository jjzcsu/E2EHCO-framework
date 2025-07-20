import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
from collections import deque
import random
import math

class ReplayBuffer:
    """高级经验回放缓冲区，支持优先级采样和多步学习"""
    
    def __init__(self, state_dim, action_dim, buffer_size=1000000, 
                 n_step=3, gamma=0.99, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        
        # 存储数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool)
        
        # 优先级
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        
        # 索引管理
        self.ptr = 0
        self.size = 0
        
        # N步缓冲区
        self.n_step_buffer = deque(maxlen=n_step)
        
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        # 存储到N步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # 计算N步回报
        n_step_reward = 0
        n_step_next_state = next_state
        n_step_done = done
        
        for i in range(self.n_step):
            n_step_reward += (self.gamma ** i) * self.n_step_buffer[-(i+1)][2]
            if self.n_step_buffer[-(i+1)][4]:  # done
                n_step_next_state = self.n_step_buffer[-(i+1)][3]
                n_step_done = True
                break
                
        # 存储N步经验
        self.states[self.ptr] = self.n_step_buffer[0][0]
        self.actions[self.ptr] = self.n_step_buffer[0][1]
        self.rewards[self.ptr] = n_step_reward
        self.next_states[self.ptr] = n_step_next_state
        self.dones[self.ptr] = n_step_done
        
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
            self.priorities[idx] = priority + 1e-6  # 避免优先级为0
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size

class GaussianPolicy(Model):
    """高斯策略网络，支持自动调节温度参数"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 activation='relu', log_std_min=-20, log_std_max=2,
                 use_batch_norm=True, use_dropout=True, dropout_rate=0.1):
        super(GaussianPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 网络层
        self.layers_list = []
        for i, dim in enumerate(hidden_dims):
            self.layers_list.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.layers_list.append(layers.BatchNormalization())
            if use_dropout and i < len(hidden_dims) - 1:
                self.layers_list.append(layers.Dropout(dropout_rate))
                
        # 输出层
        self.mean_layer = layers.Dense(action_dim, activation='tanh')
        self.log_std_layer = layers.Dense(action_dim)
        
        # 温度参数（可学习）
        self.log_alpha = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        
    def call(self, states, training=None):
        x = states
        for layer in self.layers_list:
            x = layer(x, training=training)
            
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def get_action_and_log_prob(self, states, training=None):
        """获取动作和对数概率"""
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
        
    def get_log_prob(self, states, actions, training=None):
        """计算给定动作的对数概率"""
        mean, log_std = self(states, training=training)
        std = tf.exp(log_std)
        
        normal = tfp.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions)
        log_probs = log_probs - tf.reduce_sum(tf.math.log(1 - tf.tanh(actions)**2 + 1e-6), axis=-1, keepdims=True)
        
        return log_probs

class QNetwork(Model):
    """Q网络，支持双Q网络架构"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 activation='relu', use_batch_norm=True, use_dropout=True, dropout_rate=0.1):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态编码器
        self.state_layers = []
        for dim in hidden_dims[:-1]:
            self.state_layers.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.state_layers.append(layers.BatchNormalization())
            if use_dropout:
                self.state_layers.append(layers.Dropout(dropout_rate))
                
        # 动作编码器
        self.action_layers = []
        for dim in hidden_dims[:-1]:
            self.action_layers.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.state_layers.append(layers.BatchNormalization())
            if use_dropout:
                self.state_layers.append(layers.Dropout(dropout_rate))
                
        # 合并层
        self.merge_layers = []
        for dim in hidden_dims:
            self.merge_layers.append(layers.Dense(dim, activation=activation))
            if use_batch_norm:
                self.merge_layers.append(layers.BatchNormalization())
            if use_dropout:
                self.merge_layers.append(layers.Dropout(dropout_rate))
                
        # 输出层
        self.output_layer = layers.Dense(1)
        
    def call(self, states, actions, training=None):
        # 状态编码
        state_features = states
        for layer in self.state_layers:
            state_features = layer(state_features, training=training)
            
        # 动作编码
        action_features = actions
        for layer in self.action_layers:
            action_features = layer(action_features, training=training)
            
        # 合并特征
        merged = tf.concat([state_features, action_features], axis=-1)
        
        # 处理合并特征
        for layer in self.merge_layers:
            merged = layer(merged, training=training)
            
        # 输出Q值
        q_value = self.output_layer(merged)
        
        return q_value

class SACAgent:
    """高级SAC智能体，包含多种优化技巧"""
    
    def __init__(self, state_dim, action_dim, 
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True,
                 buffer_size=1000000, batch_size=256,
                 hidden_dims=[256, 256], n_step=3,
                 use_priority_replay=True, use_double_q=True,
                 use_target_policy_smoothing=True, noise_clip=0.5, noise_sigma=0.2):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.batch_size = batch_size
        self.use_double_q = use_double_q
        self.use_target_policy_smoothing = use_target_policy_smoothing
        self.noise_clip = noise_clip
        self.noise_sigma = noise_sigma
        
        # 创建网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
        
        # 目标网络
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dims)
        
        # 复制权重到目标网络
        self.target_q1.set_weights(self.q1.get_weights())
        self.target_q2.set_weights(self.q2.get_weights())
        
        # 优化器
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.q1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.q2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_alpha)
        
        # 经验缓冲区
        self.replay_buffer = ReplayBuffer(
            state_dim, action_dim, buffer_size, n_step, gamma,
            alpha=0.6 if use_priority_replay else 0.0,
            beta=0.4 if use_priority_replay else 0.0
        )
        
        # 训练统计
        self.train_stats = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'q1_value': [],
            'q2_value': []
        }
        
    @tf.function
    def train_step(self, batch):
        """单步训练"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        weights = batch.get('weights', tf.ones(self.batch_size))
        
        with tf.GradientTape(persistent=True) as tape:
            # 当前Q值
            current_q1 = self.q1(states, actions, training=True)
            current_q2 = self.q2(states, actions, training=True)
            
            # 目标Q值计算
            next_actions, next_log_probs = self.policy.get_action_and_log_prob(next_states, training=True)
            
            # 目标策略平滑（可选）
            if self.use_target_policy_smoothing:
                noise = tf.random.normal(next_actions.shape, 0, self.noise_sigma)
                noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
                next_actions = tf.clip_by_value(next_actions + noise, -1, 1)
                
            # 目标Q值
            target_q1 = self.target_q1(next_states, next_actions, training=False)
            target_q2 = self.target_q2(next_states, next_actions, training=False)
            
            # 双Q网络：取最小值
            if self.use_double_q:
                target_q = tf.minimum(target_q1, target_q2)
            else:
                target_q = target_q1
                
            # 计算目标
            alpha = tf.exp(self.policy.log_alpha)
            target_q = target_q - alpha * next_log_probs
            target_q = rewards + self.gamma * target_q * (1 - dones)
            
            # Q网络损失
            q1_loss = tf.reduce_mean(weights * tf.square(target_q - current_q1))
            q2_loss = tf.reduce_mean(weights * tf.square(target_q - current_q2))
            
            # 策略损失
            new_actions, new_log_probs = self.policy.get_action_and_log_prob(states, training=True)
            new_q1 = self.q1(states, new_actions, training=True)
            new_q2 = self.q2(states, new_actions, training=True)
            
            if self.use_double_q:
                new_q = tf.minimum(new_q1, new_q2)
            else:
                new_q = new_q1
                
            policy_loss = tf.reduce_mean(alpha * new_log_probs - new_q)
            
            # 温度参数损失（自动调节）
            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(alpha * (new_log_probs + self.alpha))
            else:
                alpha_loss = 0.0
                
        # 计算梯度
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        q1_grads = tape.gradient(q1_loss, self.q1.trainable_variables)
        q2_grads = tape.gradient(q2_loss, self.q2.trainable_variables)
        
        if self.auto_alpha:
            alpha_grads = tape.gradient(alpha_loss, [self.policy.log_alpha])
            
        # 应用梯度
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        self.q1_optimizer.apply_gradients(zip(q1_grads, self.q1.trainable_variables))
        self.q2_optimizer.apply_gradients(zip(q2_grads, self.q2.trainable_variables))
        
        if self.auto_alpha:
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.policy.log_alpha]))
            
        # 软更新目标网络
        self._soft_update_targets()
        
        return {
            'policy_loss': policy_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'alpha_loss': alpha_loss if self.auto_alpha else 0.0,
            'alpha_value': tf.exp(self.policy.log_alpha),
            'q1_value': tf.reduce_mean(current_q1),
            'q2_value': tf.reduce_mean(current_q2)
        }
        
    def _soft_update_targets(self):
        """软更新目标网络"""
        for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
            for target_param, source_param in zip(target.trainable_variables, source.trainable_variables):
                target_param.assign((1 - self.tau) * target_param + self.tau * source_param)
                
    def train(self):
        """训练智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        stats = self.train_step(batch)
        
        # 更新优先级（如果使用优先级回放）
        if hasattr(self.replay_buffer, 'update_priorities'):
            # 这里简化处理，实际应该基于TD误差更新优先级
            priorities = np.abs(stats['q1_value'].numpy()) + np.abs(stats['q2_value'].numpy())
            self.replay_buffer.update_priorities(batch['indices'], priorities)
            
        # 记录统计信息
        for key, value in stats.items():
            self.train_stats[key].append(float(value))
            
    def get_action(self, state, training=True):
        """获取动作"""
        state = tf.expand_dims(state, 0)
        
        if training:
            actions, _ = self.policy.get_action_and_log_prob(state, training=True)
            return actions[0].numpy()
        else:
            mean, _ = self.policy(state, training=False)
            return tf.tanh(mean[0]).numpy()
            
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def save_model(self, filepath):
        """保存模型"""
        self.policy.save_weights(filepath + '_policy')
        self.q1.save_weights(filepath + '_q1')
        self.q2.save_weights(filepath + '_q2')
        
    def load_model(self, filepath):
        """加载模型"""
        self.policy.load_weights(filepath + '_policy')
        self.q1.load_weights(filepath + '_q1')
        self.q2.load_weights(filepath + '_q2')
        self.target_q1.set_weights(self.q1.get_weights())
        self.target_q2.set_weights(self.q2.get_weights())
        
    def get_training_stats(self):
        """获取训练统计信息"""
        return {key: np.mean(values[-100:]) if values else 0 
                for key, values in self.train_stats.items()}

class AdaptiveSAC(SACAgent):
    """自适应SAC，动态调整超参数"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自适应参数
        self.adaptive_params = {
            'alpha': self.alpha,
            'tau': self.tau,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic
        }
        
        # 性能跟踪
        self.performance_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        
    def adapt_hyperparameters(self, recent_rewards, recent_entropies):
        """自适应调整超参数"""
        if len(recent_rewards) < 20:
            return
            
        # 计算性能指标
        mean_reward = np.mean(recent_rewards)
        mean_entropy = np.mean(recent_entropies)
        
        # 更新历史
        self.performance_history.append(mean_reward)
        self.entropy_history.append(mean_entropy)
        
        if len(self.performance_history) < 50:
            return
            
        # 计算趋势
        recent_performance = np.mean(list(self.performance_history)[-20:])
        old_performance = np.mean(list(self.performance_history)[-50:-20])
        
        # 根据性能趋势调整参数
        if recent_performance > old_performance * 1.05:  # 性能提升
            # 增加探索
            self.adaptive_params['alpha'] = min(0.5, self.adaptive_params['alpha'] * 1.02)
            self.adaptive_params['tau'] = min(0.01, self.adaptive_params['tau'] * 1.01)
        elif recent_performance < old_performance * 0.95:  # 性能下降
            # 减少探索
            self.adaptive_params['alpha'] = max(0.05, self.adaptive_params['alpha'] * 0.98)
            self.adaptive_params['tau'] = max(0.001, self.adaptive_params['tau'] * 0.99)
            
        # 根据熵调整学习率
        if mean_entropy < 0.1:  # 熵太低，增加学习率
            self.adaptive_params['lr_actor'] *= 1.01
            self.adaptive_params['lr_critic'] *= 1.01
        elif mean_entropy > 2.0:  # 熵太高，降低学习率
            self.adaptive_params['lr_actor'] *= 0.99
            self.adaptive_params['lr_critic'] *= 0.99
            
        # 应用调整后的参数
        self.alpha = self.adaptive_params['alpha']
        self.tau = self.adaptive_params['tau']
        
        # 更新优化器学习率
        self.policy_optimizer.learning_rate.assign(self.adaptive_params['lr_actor'])
        self.q1_optimizer.learning_rate.assign(self.adaptive_params['lr_critic'])
        self.q2_optimizer.learning_rate.assign(self.adaptive_params['lr_critic']) 