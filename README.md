# E2EHCO Framework: A Dynamic Edge Computing Task Offloading Solution

## Project Background
Mobile Edge Computing (MEC) deploys computational resources at the network edge, enabling task offloading for compute-intensive applications. However, in dynamic multi-user, multi-server environments, user mobility leads to time-varying channel conditions, and the spatiotemporal heterogeneity of server loads further complicates system behavior. The system must jointly optimize discrete offloading decisions and continuous resource-allocation parameters, forming a hybrid action space. Its integrated decision-making mechanism is key to breaking the long-standing trade-off between latency and energy consumption.

## Technical Challenges
Traditional deep reinforcement learning (DRL) approaches relying on a single policy network often suffer from strong strategy coupling and Q-value estimation bias. This results in policy oscillations and the curse of dimensionality in highly dynamic scenarios, hindering stable convergence.

## Solution
This project proposes an End-to-End Hybrid Computation Offloading (E2EHCO) framework based on an enhanced Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm:
- Employs dual critic networks with a delayed-update mechanism to effectively suppress Q-value overestimation
- Integrates Softmax and Tanh activations in the actor network to handle discrete and continuous actions simultaneously
- Achieves efficient and robust joint decision optimization in dynamically changing conditions

## Performance Advantages
Experiments on real-world mobility traces demonstrate that in high-density user scenarios:
- Reduces total latency by at least 20% compared to benchmark methods
- Lowers energy consumption by approximately 16%
- Provides an adaptive offloading solution with real-time responsiveness for large-scale, dynamic MEC systems

![边缘计算架构](image/Summary.png)





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

