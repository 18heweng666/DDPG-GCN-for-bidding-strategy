import os

import numpy as np
import matplotlib.pyplot as plt

from market.multi_choice_bus import *


# def smooth(data, weight=0.9):
#     '''用于平滑曲线，类似于Tensorboard中的smooth曲线
#     '''
#     last = data[0]
#     smoothed = []
#     for point in data:
#         smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     return smoothed


import numpy as np
import pandas as pd


def disconnect_branch(mpc, branch_id):
    mpc1 = mpc.copy()
    for i in range(len(branch_id)):
        mpc1['branch'][branch_id[i]][10] = 0
    return mpc1


def disconnect_bus(mpc, bus_id):
    mpc1 = mpc.copy()
    for i in range(len(bus_id)):
        mpc1['bus'][bus_id[i]][1] = 4
    return mpc1


def disconnect_gen(mpc, gen_id, n_agents_gen):
    mpc1 = mpc.copy()
    for i in range(len(gen_id)):
        gen_type = 'gen' if gen_id[i] < n_agents_gen else 'load'
        # mpc1['gen'][gen_id[i]][7] = 0
        # mpc1['gen'][gen_id[i]][8:] = [0,0]
        if gen_type == 'gen':
            mpc1['gen'][gen_id[i]][8] = 1e-8
        elif gen_type == 'load':
            mpc1['gen'][gen_id[i]][9] = -1e-8
    return mpc1


def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    smoothed = np.zeros_like(data)  # 创建与输入数据相同形状的零矩阵
    last = data[0, :]  # 获取每个序列的初始值
    smoothed[0, :] = last  # 平滑后的第一个点与初始值相同

    for t in range(1, data.shape[0]):
        smoothed[t, :] = last * weight + (1 - weight) * data[t, :]  # 计算平滑值
        last = smoothed[t, :]  # 更新last为当前平滑值

    return smoothed


def linear_map(x, in_min, in_max, out_min, out_max): #若想实现大值映射小值，小值映射大值，可以直接将输入的out_min和out_max调换，如0.1和0.01
    """
    线性映射函数
    将输入值 x 映射到 [out_min, out_max] 区间内的值
    假设输入值 x 在 [in_min, in_max] 区间内
    """
    # 将输入值 x 映射到 [0, 1] 区间内
    x_normalized = (x - in_min) / (in_max - in_min)
    # 将 [0, 1] 区间内的值映射到 [out_min, out_max] 区间内
    max_true = max(out_min,out_max)
    min_true = min(out_min,out_max)
    result = max(min(out_min + (x_normalized * (out_max - out_min)), max_true), min_true) #不知道为什么有时候会溢出，所以加一个这个
    return result


def inverse_linear_map(y, in_min, in_max, out_min, out_max):
    """
    反线性映射函数
    将输出值 y 反映射回 [in_min, in_max] 区间内的值
    假设输出值 y 在 [out_min, out_max] 区间内
    """
    # 将输出值 y 映射回 [0, 1] 区间内
    y_normalized = (y - out_min) / (out_max - out_min)
    # 将 [0, 1] 区间内的值反映射回 [in_min, in_max] 区间内
    result = max(min(in_min + (y_normalized * (in_max - in_min)), in_max), in_min)
    return result



def reset_env(gen_low,gen_high,load_low,load_high,n_agents,n_agents_gen):
    # s_1 = np.random.uniform(gen_low, gen_high, n_agents_gen)
    # s_2 = np.random.uniform(load_low, load_high, n_agents - n_agents_gen)
    # return np.concatenate((s_1, s_2))

    s_1 = np.random.uniform(20, 60, n_agents_gen)
    s_2 = np.random.uniform(20, 60, n_agents - n_agents_gen)
    return np.concatenate((s_1, s_2))


def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if cfg.save_fig:
        plt.savefig(cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path+'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")


def visualize_sequence_data_compare(path_list):
    algo_num = len(path_list)
    data = pd.read_csv(os.path.join(path_list[0], 'average_episodic_rewards.csv')).values
    data_std = pd.read_csv(os.path.join(path_list[0], 'average_episodic_rewards_std.csv')).values

    for i in range(algo_num-1):
        result_path_i = path_list[i+1]
        data_i = pd.read_csv(os.path.join(result_path_i,'average_episodic_rewards.csv')).values
        data_std_i = pd.read_csv(os.path.join(result_path_i,'average_episodic_rewards_std.csv')).values

        if len(data_i) > len(data):
            data_i = data_i.values[:-1]
            data_std_i = data_std_i.values[:-1]

        data = np.concatenate((data, data_i), axis=1)  #[memory_episode:]
        # data = data[memory_episode:, :] if not show_memory else data
        data_std = np.concatenate((data_std, data_std_i), axis=1)
        # data_std = data_std[memory_episode:, :] if not show_memory else data_std
    return data, data_std





def visualize_sequence_data(data, labels=None, color=None, xlabel='t', ylabel='value', filename=None, dpi=300, test_step=None, show_memory_step=False, show_test_step=True, memory_step=None, subplot=False, show_legend=True, fill_std=None, std_scale=0.4, xtick=None, errorbar=False, xlim=None, figsize=None, cmap='viridis'):
    """
    Visualize sequence data using different colors for each sequence and show labels.

    Args:
        data (numpy array): A 2D numpy array of shape (num_timesteps, num_sequences) containing the sequence data.
        labels (list): A list of strings containing the labels for each sequence.
        filename (str): Optional. If specified, save the plot to this filename instead of displaying it.
    """
    if subplot is False:
        plt.figure(figsize=figsize)
    # Get the number of sequences and timesteps
    num_timesteps, num_sequences = data.shape


    if not color:
        cmap = plt.get_cmap(cmap) # Define a color map to use for each sequence viridis
        colors = cmap(np.linspace(0, 1, num_sequences))  # (num_sequences,4)ndarray,最后一列都是1，可能没什么实际意义
    else:
        colors = color

    # Plot each sequence using a different color
    for i in range(num_sequences):
        if labels is not None:
            plt.plot(data[:, i], color=colors[i], label=labels[i]) #if show_memory_step else data[memory_step:, i]
        else:
            plt.plot(data[:, i], color=colors[i])

        if fill_std is not None:
            if errorbar is False:
                plt.fill_between(range(len(data[:, i])), data[:, i] - fill_std[:,i]*std_scale, data[:, i] + fill_std[:,i]*std_scale, alpha=0.1, color=colors[i])
            else:
                plt.errorbar(np.arange(len(data[:, i])), data[:, i], yerr=fill_std[:,i], fmt='-o', capsize=4, color=colors[i], alpha=0.5) # 绘制折线图和误差线


    if show_legend:
        if num_sequences > 1 and labels is not None:
            plt.legend()
            # plt.legend(loc='lower right')
            # plt.legend(bbox_to_anchor=(0.40, 0.5))

    if show_test_step and test_step is not None:
        plt.axvline(x=num_timesteps-test_step, color='black', linestyle='dashed', alpha=0.3, linewidth=1)

    if show_memory_step and memory_step is not None:
        plt.axvline(x=memory_step, color='black', linestyle='dashed', alpha=0.3, linewidth=1)


    # Add axis labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title('Sequence Data')
    # plt.xticks(np.arange(num_timesteps), xtick)  #GAMMA画图可开，其他记得关，否则严重影响效率！！！
    plt.autoscale(axis='x', tight=True)
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(*xlim)

    # Save or show the plot
    if subplot is False:
        if filename:
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()

def add_rows_as_columns(x, n):
    rows, cols = x.shape
    new_cols = cols * n
    result = np.zeros((rows, new_cols))

    for i in range(rows):
        for j in range(n):
            if i + j < rows:
                result[i, j * cols:(j + 1) * cols] = x[i + j]
            # else:
            #     # 如果超出范围，用零填充
            #     result[i, j * cols:(j + 1) * cols] = np.zeros(cols)

    return result


if __name__ == '__main__':
    #E94849 #5D93BF #71BE6E
    color = ['#E94849', '#5D93BF', '#71BE6E', '#F4E478']
    color = ['#FF0000', '#0000FF', '#00FF00', '#939294']
    # Generate some sample data
    data = np.random.randn(100, 4)
    labels = ['gen 1', 'gen 2', 'gen3', 'gen4']
    # Visualize the data
    visualize_sequence_data(data, labels, color)