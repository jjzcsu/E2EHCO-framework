"""
强化学习算法集合
包含多种经典和先进的强化学习算法实现
"""

from .ppo import PPOAgent, AdaptivePPO
from .sac import SACAgent, AdaptiveSAC
from .dqn import DQNAgent, AdaptiveDQN
from .a2c import A2CAgent
from .trpo import TRPOAgent

__all__ = [
    'PPOAgent',
    'AdaptivePPO', 
    'SACAgent',
    'AdaptiveSAC',
    'DQNAgent',
    'AdaptiveDQN',
    'A2CAgent',
    'TRPOAgent'
]

class RLAlgorithmManager:
    """强化学习算法管理器"""
    
    def __init__(self):
        self.algorithms = {
            'ppo': PPOAgent,
            'adaptive_ppo': AdaptivePPO,
            'sac': SACAgent,
            'adaptive_sac': AdaptiveSAC,
            'dqn': DQNAgent,
            'adaptive_dqn': AdaptiveDQN,
            'a2c': A2CAgent,
            'trpo': TRPOAgent
        }
        
    def create_agent(self, algorithm_name, **kwargs):
        """创建智能体"""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
            
        return self.algorithms[algorithm_name](**kwargs)
        
    def get_available_algorithms(self):
        """获取可用的算法列表"""
        return list(self.algorithms.keys())
        
    def get_algorithm_info(self, algorithm_name):
        """获取算法信息"""
        if algorithm_name not in self.algorithms:
            return None
            
        algorithm = self.algorithms[algorithm_name]
        
        info = {
            'name': algorithm_name,
            'class': algorithm.__name__,
            'description': algorithm.__doc__ or '无描述',
            'parameters': algorithm.__init__.__code__.co_varnames[1:algorithm.__init__.__code__.co_argcount]
        }
        
        return info 