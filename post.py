import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(file_path):
    # 1. 读取数据
    # sep='\t' 对应你的文件是以 Tab 分割的
    # comment='#' 会自动过滤掉文件头部的超参数部分
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 提取列数据
    episodes = df['Episode']
    survival_time = df['Current_Lifespan']
    
    # 自动获取总时间（取最后一行的 Total_Elapsed_Time(s)）
    total_time = df['Total_Elapsed_Time(s)'].iloc[-1]

    # 2. 计算滑动平均和滑动标准差
    window_size = 50
    # center=False 模拟实时训练的平滑效果，True 则更对称
    surv_mean = survival_time.rolling(window=window_size, min_periods=1).mean()
    surv_std = survival_time.rolling(window=window_size, min_periods=1).std()

    # 3. 绘图
    plt.figure(figsize=(12, 9), dpi=100)
    
    # 绘制原始数据（灰色细线）
    plt.plot(episodes, survival_time, color='lightgray', linewidth=1, label='Raw Survival Time')

    # 绘制滑动平均主曲线
    plt.plot(episodes, surv_mean, color='#1f77b4', linewidth=2, label=f'Sliding Mean (W={window_size})')

    # 绘制 ±1 标准差的阴影区域
    plt.fill_between(episodes, 
                     surv_mean - surv_std, 
                     surv_mean + surv_std, 
                     color='#1f77b4', alpha=0.2, label='Std Deviation')

    # 4. 图表修饰
    plt.title(f'Cartpole-v1 Training Process (Total Time: {total_time:.1f}s)', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Survival Time', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    
    # 自动紧凑布局
    plt.savefig('training_result.svg', format='svg', bbox_inches='tight')

    print("文件已保存为 SVG 格式，请直接用浏览器打开查看。")

# 调用函数
if __name__ == "__main__":
    plot_training_results('dqn_results1185.txt') # 确保文件名一致