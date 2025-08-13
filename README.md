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
│   ├── td3_mec.py         
│   ├── mec.py              
│   ├── env.py              
│   └── model/              
│       ├── ppo.py         
│       ├── sac.py          
│       ├── dqn.py          
│       ├── a2c.py          
│       └── trpo.py        
├── data/                  
├── output/                 
└── image/                  
```

