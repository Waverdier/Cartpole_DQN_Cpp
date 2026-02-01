import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time

"""
Instructions
All Parameter including output file name
could be modified in main function-hyperparameters
"""

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 4 -> 128 -> 128 -> 2 (CartPole的状态维度是4，动作维度是2)
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x) #了解
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 存储一个元组 (state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从缓冲区随机抽取一个批次
        transitions = random.sample(self.buffer, batch_size)
        # 将批次中的元素解包并堆叠成 NumPy 数组
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)
    
# 核心训练函数 (简化版)
def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return

    # 1. 采样 (Sample)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

    # 转换为 PyTorch Tensor
    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1) # [B, 1]
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)
    
    # 2. 计算当前 Q(s, a) (Current Q Value)
    # policy_net(state_batch) 输出 [B, 2]，.gather(1, action_batch) 选取实际执行动作的 Q 值
    current_q_values = policy_net(state_batch).gather(1, action_batch) 

    # 3. 计算目标 Q 值 (Target Q Value)
    # target_net 预测下一状态的 Q 值，.max(1) 找出最大值
    next_state_q_values = target_net(next_state_batch).max(1)[0].detach()
    # 贝尔曼方程：r + gamma * max(Q(s', a')) * (1 - done)
    # (1 - done) 确保在终止状态时，Q值直接为 r
    target_q_values = reward_batch + (gamma * next_state_q_values * (1 - done_batch))
    target_q_values = target_q_values.unsqueeze(1) # [B, 1]

    # 4. 计算损失和优化 (Loss and Optimization)
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10) # 可选的梯度裁剪
    optimizer.step()

#Visualization
def plot_durations():
    plt.figure(2)
    plt.title('Lifespan')
    plt.xlabel('Episode')
    plt.ylabel('Lifespan')
    
    # 原始数据
    plt.plot(episode_durations, label='Lifespan vs Espisode')

    # 计算移动平均值 (例如 100 个 episode 的平均)
    if len(episode_durations) >= PLOT_WINDOW:
        # 使用 numpy 或 pandas 计算更方便。这里用简单的 PyTorch/Python 列表操作
        means = np.array([np.mean(episode_durations[i-PLOT_WINDOW:i]) 
                          for i in range(PLOT_WINDOW, len(episode_durations) + 1)])
        plt.plot(np.arange(PLOT_WINDOW, len(episode_durations) + 1), means, 
                 label=f'{PLOT_WINDOW} Episode Avg', color='red', linewidth=2)
    
    plt.legend()
    # 如果在 notebook 中，可以使用 IPython.display 相关的刷新命令

#TXT Output
def save_results_to_txt(episode_data, episode_times, params, filename):
    """
    将超参数、episode 时长和物理时间保存到 TXT 文件中。
    
    Args:
        episode_data (list): 包含每个 episode 时长的列表。
        episode_times (list): 包含每个 episode 物理耗时的列表。
        params (dict): 包含所有超参数的字典。
        filename (str): 输出文件名。
    """
    print(f"\nSaving results to {filename}...")
    with open(filename, 'w') as f:
        
        # --- 写入超参数头部 ---
        f.write("# --- Hyperparameters ---\n")
        for key, value in params.items():
            f.write(f"# {key}: {value}\n")
        f.write("# -----------------------\n\n")
        
        # --- 写入数据列头 ---
        f.write("Episode\tLifespan (t+1)\tPhysical Time (s)\n") 
        
        # --- 写入每一行数据 ---
        for i, (duration, time_spent) in enumerate(zip(episode_data, episode_times)):
            f.write(f"{i+1}\t{duration}\t{time_spent:.4f}\n")
            
    print("Saving complete.")
# ------------------------------------

# --- 超参数设置 ---
GAMMA = 0.99           # 折扣因子
EPS_START = 1.0        # 初始探索率
EPS_END = 0.01         # 最终探索率
EPS_DECAY = 200        # 探索率衰减速率
TARGET_UPDATE = 10     # 目标网络更新频率（每多少步）
MEMORY_CAPACITY = 10000 # 经验回放容量
BATCH_SIZE = 64        # 批次大小
LR = 1e-4              #Learning Rate
EPISODE_SETTINGS = 1500 #Total Episodes
F_NAME ="dqn_cartpole_results.txt"

# 收集所有超参数到一个字典中，方便传递
HYPER_PARAMS = {
    "GAMMA": GAMMA,
    "EPS_START": EPS_START,
    "EPS_END": EPS_END,
    "EPS_DECAY": EPS_DECAY,
    "TARGET_UPDATE": TARGET_UPDATE,
    "MEMORY_CAPACITY": MEMORY_CAPACITY,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "Total Episodes": EPISODE_SETTINGS
}

# --- 初始化 ---
env = gym.make("CartPole-v1") #, render_mode="human" for visulizaion(put in into the bracket)
state_dim = env.observation_space.shape[0] # 4
action_dim = env.action_space.n           # 2

policy_net = DQN(state_dim, action_dim) # 策略网络
target_net = DQN(state_dim, action_dim) # 目标网络
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # 目标网络设置为评估模式，不训练

optimizer = optim.AdamW(policy_net.parameters(), lr= LR)
memory = ReplayBuffer(MEMORY_CAPACITY)

steps_done = 0

# --- 初始化可视化存储列表 ---
episode_durations = [] # 存储每个 episode 的步数/时长
episode_times = [] #Time for every episode

# 设置运行平均窗口，用于平滑曲线
PLOT_WINDOW = 100

# --- 训练循环 ---
for episode in range(EPISODE_SETTINGS):
    state, _ = env.reset() # 初始化状态
    state = state.astype(np.float32) # 转换为float32
    start_time = time.time()#Note Start Time
    
    for t in range(500): # 最多运行500步
        # 1. 选择动作
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        if random.random() < epsilon:
            action = env.action_space.sample() # 探索
        else:
            with torch.no_grad():
                # 评估：根据策略网络选择最优动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_tensor).max(1)[1].item() # max(1)[1] 返回最大Q值的索引

        # 2. 与环境交互
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = next_state.astype(np.float32)
        done = terminated or truncated # 任何一个为真都表示回合结束
        
        #env.render() for visualization

        # 3. 存储经验
        memory.push(state, action, reward, next_state, done)
        
        # 4. 更新状态和步数
        state = next_state
        steps_done += 1

        # 5. 优化模型
        optimize_model(policy_net, target_net, optimizer, memory, BATCH_SIZE, GAMMA)
        
        if done:
            print(f"Episode {episode} finished after {t+1} timesteps.")
            break

    # --- Episode 结束后的数据处理 ---
    end_time = time.time() # <-- 4. 记录 Episode 结束时间
    time_spent = end_time - start_time # <-- 5. 计算耗时

    episode_durations.append(t+1)
    episode_times.append(time_spent) # <-- 6. 存储耗时
        
    # 6. 目标网络更新
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

save_results_to_txt(episode_durations, episode_times, HYPER_PARAMS, F_NAME)
plot_durations()
plt.show(block=True)

env.close()