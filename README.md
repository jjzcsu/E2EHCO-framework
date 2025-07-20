# 基于TD3算法的边缘计算资源分配优化

## 项目概述

本项目提出了一种基于**Twin Delayed Deep Deterministic Policy Gradient (TD3)**算法的边缘计算资源分配优化方案。针对移动边缘计算(MEC)环境中的计算卸载、资源分配和任务调度问题，我们设计了一个智能化的资源管理系统，能够动态优化计算资源分配、带宽分配和卸载决策，以最小化系统总延迟和能耗。

![边缘计算架构](image/Summary.png)

## 核心创新

### 🚀 **TD3算法在边缘计算中的应用**
- **首次将TD3算法**应用于边缘计算资源分配问题
- **目标策略平滑**：减少Q值估计偏差，提高训练稳定性
- **延迟策略更新**：降低策略更新频率，避免过度拟合
- **双Q网络架构**：使用两个独立的Critic网络，取最小值作为目标Q值

### 🎯 **多目标优化设计**
- **延迟最小化**：优化任务传输和计算延迟
- **能耗优化**：平衡传输能耗和计算能耗
- **资源利用率最大化**：动态分配边缘服务器计算资源

## 系统架构

### 边缘计算环境

#### 移动用户 (Mobile Users)
- **用户数量**：10个移动用户
- **移动模式**：基于KAIST数据集的真实移动轨迹
- **任务类型**：VOC SSD300目标检测任务
- **任务参数**：
  - 传输数据大小：2.7 × 10⁴ bytes
  - 处理数据大小：1.08 × 10⁶ bytes
  - 返回数据大小：96 bytes

#### 边缘服务器 (Edge Servers)
- **服务器数量**：10个边缘服务器
- **计算能力**：6.3 × 10⁷ byte/sec
- **带宽容量**：1 × 10⁹ byte/sec
- **服务限制**：每个服务器最多服务4个用户

### TD3智能体设计

#### 状态空间 (State Space)
```python
# 状态维度：user_num * 2 + 3 = 23
state = [
    # 边缘服务器可用资源 (10维)
    edge_resources,
    # 可用带宽 (10维)  
    available_bandwidth,
    # 用户卸载目标 (10维)
    offloading_targets,
    # 用户位置 (20维)
    user_locations,
    # 系统性能指标 (3维)
    [delay, energy, cost]
]
```

#### 动作空间 (Action Space)
```python
# 动作维度：user_num * (edge_num + 1) + user_num + edge_num * user_num = 130
action = [
    # 卸载决策 (110维)
    offloading_decisions,
    # 传输功率 (10维)
    transmission_power,
    # 计算资源分配 (100维)
    resource_allocation
]
```

#### 奖励函数 (Reward Function)
```python
reward = 1 / cost
# cost = w1 * delay + w2 * energy
# 其中 w1 = w2 = 0.5
```

## 算法实现

### TD3网络架构

#### Actor网络 (策略网络)
```python
def get_actor():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)
    return model
```

#### Critic网络 (价值网络)
```python
def get_critic():
    state_input = layers.Input(shape=(num_states))
    action_input = layers.Input(shape=(num_actions))
    state_out = layers.Dense(400, activation="relu")(state_input)
    action_out = layers.Dense(400, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(300, activation="relu")(concat)
    outputs = layers.Dense(1)(out)
    return model
```

### TD3核心特性

#### 1. 目标策略平滑 (Target Policy Smoothing)
```python
# 在目标动作上添加噪声
target_actions = target_actor(next_state_batch, training=True)
noise = tf.random.normal(target_actions.shape, 0, 0.2)
noise = tf.clip_by_value(noise, -0.5, 0.5)
target_actions = tf.clip_by_value(target_actions + noise, -1, 1)
```

#### 2. 延迟策略更新 (Delayed Policy Updates)
```python
# 每2步更新一次策略网络
if self.buffer_counter % 2 == 0:
    # 更新Actor网络
    actor_loss = -tf.math.reduce_mean(critic_value)
```

#### 3. 双Q网络 (Twin Q-Networks)
```python
# 使用两个独立的Critic网络
y1 = reward_batch + gamma * target_1_critic([next_state_batch, target_actions])
y2 = reward_batch + gamma * target_2_critic([next_state_batch, target_actions])
min_q_target = tf.minimum(y1, y2)
```

## 实验设置

### 训练参数
- **总训练轮数**：500 episodes
- **每轮步数**：3000 steps
- **学习率**：Actor=3e-4, Critic=3e-4
- **折扣因子**：γ = 0.99
- **目标网络更新率**：τ = 0.005
- **经验缓冲区**：100,000容量
- **批次大小**：256

### 环境参数
- **用户数量**：10个
- **边缘服务器数量**：10个
- **带宽**：1 × 10⁹ byte/sec
- **计算能力**：6.3 × 10⁷ byte/sec
- **传输功率**：500 mW
- **闲时功率**：100 mW

## 运行指南

### 环境要求
```bash
Python 3.7.5+
TensorFlow 2.2.0+
NumPy
Matplotlib
TensorboardX
```

### 运行TD3算法
```bash
# 运行TD3训练
python src/td3_mec.py

# 查看训练日志
tensorboard --logdir=tensorboard_data
```

### 参数配置
```python
# 在td3_mec.py中修改参数
user_num = 10          # 用户数量
edge_num = 10          # 边缘服务器数量
total_episodes = 500   # 训练轮数
buffer_size = 100000   # 缓冲区大小
batch_size = 256       # 批次大小
```

## 性能评估

### 评估指标
1. **系统总延迟**：任务传输和计算的总时间
2. **系统总能耗**：传输和计算的总能耗
3. **资源利用率**：边缘服务器计算资源的利用效率
4. **任务完成率**：成功完成的任务比例

### 对比基准
- **DDPG算法**：深度确定性策略梯度
- **PPO算法**：近端策略优化
- **SAC算法**：软演员评论家
- **传统启发式方法**：基于距离的最近服务器分配

## 项目结构

```
├── src/
│   ├── td3_mec.py          # TD3算法主实现
│   ├── mec.py              # 边缘计算环境
│   ├── env.py              # 原始环境实现
│   └── model/              # 强化学习算法集合
│       ├── ppo.py          # PPO算法
│       ├── sac.py          # SAC算法
│       ├── dqn.py          # DQN算法
│       ├── a2c.py          # A2C算法
│       └── trpo.py         # TRPO算法
├── data/                   # KAIST移动轨迹数据
├── output/                 # 实验结果
└── image/                  # 项目图片
```

## 主要贡献

### 1. **算法创新**
- 首次将TD3算法应用于边缘计算资源分配
- 设计了适合边缘计算环境的状态和动作空间
- 实现了多目标优化的奖励函数

### 2. **系统优化**
- 动态资源分配策略
- 智能卸载决策机制
- 实时性能监控和调整

### 3. **实验验证**
- 基于真实移动轨迹数据的仿真
- 多场景性能对比分析
- 算法收敛性和稳定性验证

## 未来工作

1. **多智能体TD3**：扩展到多智能体协作场景
2. **在线学习**：实现实时在线学习和适应
3. **异构环境**：支持不同类型的边缘设备和任务
4. **安全机制**：加入隐私保护和安全性考虑

## 参考文献

1. Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *ICML*.
2. Han, M., et al. (2008). CRAWDAD dataset kaist/wibro. *CRAWDAD*.
3. Liu, L., et al. (2016). Mobile edge computing: A survey on the hardware-software reference architecture. *ACM Computing Surveys*.

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目地址：[GitHub Repository]
- 邮箱：[your-email@example.com]

---

**注意**：本项目仅用于学术研究目的，请勿用于商业用途。
