# 基于TD3算法的边缘计算资源分配优化

## 项目概述

本项目提出了一种基于**Twin Delayed Deep Deterministic Policy Gradient (TD3)**算法的边缘计算资源分配优化方案。针对移动边缘计算(MEC)环境中的计算卸载、资源分配和任务调度问题，我们设计了一个智能化的资源管理系统，能够动态优化计算资源分配、带宽分配和卸载决策，以最小化系统总延迟和能耗。

![边缘计算架构](image/Summary.png)


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



**注意**：本项目仅用于学术研究目的，请勿用于商业用途。
