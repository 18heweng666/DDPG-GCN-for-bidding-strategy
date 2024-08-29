import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import os

def plot_experiment(result_path, algo, case_name, time_vary_data_need, agent_flag, episode_num, memory_step, smooth_rate_default = 0.96):
    action_gamma = False


    # agent_flag = [1,0,1,0, 0,1,0,1] #不影响各种csv结果保存，只影响画图
    # agent_flag = [1,0,1,1, 0,1,0,0] #不影响各种csv结果保存，只影响画图
    # agent_flag = [1,0,0,0, 0,1,0,0] #不影响各种csv结果保存，只影响画图
    # agent_flag = [1,1,1,1, 1,1,1,1] #不影响各种csv结果保存，只影响画图
    # agent_flag = [0,0,0,0, 1,1,1,1] #不影响各种csv结果保存，只影响画图
    # agent_flag = [1,1,1,1, 0,0,0,0] #不影响各种csv结果保存，只影响画图
    # agent_flag = [1,0,1,1,0,0, 1,0,0]
    # agent_flag = [1,1,1,1,0,0, 0,0,0]
    # agent_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # agent_flag = [1,1,1,1, 0,0, 0,1 ,1] #如果要测试一下[1,1,1,1, 0,0,0,0]和[1,1,1,1,1,1,1,1]的单双边情况对比，分别在两个文件夹运行一下，得到test_reward_all.csv和test_strategic_all.csv文件即可
    # agent_flag = [1,1,1,1,  1,1 ,1, 1,1,1] #如果要测试一下[1,1,1,1, 0,0,0,0]和[1,1,1,1,1,1,1,1]的单双边情况对比，分别在两个文件夹运行一下，得到test_reward_all.csv和test_strategic_all.csv文件即可
    episode_length = 24 #一局走多少步，如果没有局的概念，可以设为None

    non_zero_columns = np.where(np.array(agent_flag) == 1)[0]

    test_step = episode_length*20
    # memory_step = 1000 - 40
    # memory_episode = int(memory_step/24)-5 #int会往下取整

    memory_episode = int(memory_step/24) #int会往下取整
    plt.rcParams['font.size'] = 12
    import matplotlib as mpl
    mpl.rcParams['figure.autolayout'] = True
    show_memory = False
    show_test = True
    n_steps_experiment = int(episode_num * episode_length)
    # smooth_rate_default = 0.98  #0.95
    smooth_r = 0.96 #0.96


    # # --------------------------社会福利--------------------------------------
    #先save test strategic
    path = os.path.join(result_path,'social_welfare.csv')
    if os.path.exists(path):
        social = pd.read_csv(path)
        social = social.values
        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, social.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = social[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)
        new_list = []
        new_list1 = []

        agent_i = np.array(result_list)
        new_list.append(np.mean(agent_i,axis=0))
        new_list1.append(np.std(agent_i,axis=0))

        social_all = np.array(new_list).T
        x1 = social_all[-test_step:,:].flatten().sum(0)
        x2 = social_all[-test_step:,:].flatten().std(0) #主要用来看看这个是不是真收敛了
        # x = pd.DataFrame((x1,x2))
        # x = pd.DataFrame(social_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        # x.to_csv(os.path.join(result_path,'test_social_all.csv'),index=False,header=None)

        social_all = np.array(new_list1).T
        x3 = social_all[-test_step:,:].flatten().mean(0)
        # x2 = social_all[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
        x = pd.DataFrame((x1,x2,x3))
        # x = pd.DataFrame(social_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        x.to_csv(os.path.join(result_path,'test_social_all.csv'),index=False,header=None)
    else:
        print(f'File {path} does not exist.')


    # --------------------------load power--------------------------------------
    #先save test strategic
    path = os.path.join(result_path,'load_rate.csv')
    if os.path.exists(path):
        load_rate = pd.read_csv(path)
        load_rate = load_rate.values
        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, load_rate.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = load_rate[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)
        new_list = []
        new_list1 = []
        for i in range(load_rate.shape[1]):
            agent_i = np.array([exp[:,i] for exp in result_list])
            new_list.append(np.mean(agent_i,axis=0))
            new_list1.append(np.std(agent_i,axis=0))

        load_rate = np.array(new_list).T
        x1 = load_rate[-test_step:,:].mean(0)
        x2 = load_rate[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
        # x = pd.DataFrame((x1,x2))
        # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        # x.to_csv(os.path.join(result_path,'test_strategic_all.csv'),index=False,header=None)

        load_rate = np.array(new_list1).T
        x3 = load_rate[-test_step:,:].mean(0)
        # x2 = strategic_all[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
        x = pd.DataFrame((x1,x2,x3))
        # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        x.to_csv(os.path.join(result_path,'test_load_rate.csv'),index=False,header=None)
    else:
        print(f'File {path} does not exist.')



    # --------------------------决策系数--------------------------------------
    #先save test strategic
    strategic_variables = pd.read_csv(os.path.join(result_path,'result_action.csv'))
    strategic_variables = strategic_variables.values
    #取出每个实验的数据，存入列表
    result_list = []
    # 遍历矩阵的行
    for i in range(0, strategic_variables.shape[0], n_steps_experiment):
        # 截取每个子列表的行范围
        chunk = strategic_variables[i:i+n_steps_experiment]
        # 将子列表添加到结果列表中
        result_list.append(chunk)
    new_list = []
    new_list1 = []
    for i in range(len(agent_flag)):
        agent_i = np.array([exp[:,i] for exp in result_list])
        new_list.append(np.mean(agent_i,axis=0))
        new_list1.append(np.std(agent_i,axis=0))

    strategic_all = np.array(new_list).T
    x1 = strategic_all[-test_step:,:].mean(0)
    x2 = strategic_all[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
    # x = pd.DataFrame((x1,x2))
    # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
    # x.to_csv(os.path.join(result_path,'test_strategic_all.csv'),index=False,header=None)

    strategic_all = np.array(new_list1).T
    x3 = strategic_all[-test_step:,:].mean(0)
    # x2 = strategic_all[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
    x = pd.DataFrame((x1,x2,x3))
    # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
    x.to_csv(os.path.join(result_path,'test_strategic_all.csv'),index=False,header=None)



    #开始
    strategic_variables = pd.read_csv(os.path.join(result_path,'result_action.csv'))
    # strategic_variables = strategic_variables.values * agent_flag
    # is_zero_column = np.all(strategic_variables == 0, axis=0)
    # non_zero_columns = np.where(~is_zero_column)[0]

    strategic_variables = strategic_variables.values[:, non_zero_columns]
    strategic_variables1 = strategic_variables.copy()

    #取出每个实验的数据，存入列表
    result_list = []
    # 遍历矩阵的行
    for i in range(0, strategic_variables.shape[0], n_steps_experiment):
        # 截取每个子列表的行范围
        chunk = strategic_variables[i:i+n_steps_experiment]
        # 将子列表添加到结果列表中
        result_list.append(chunk)

    new_list = []
    new_list1 = []
    for i in range(sum(agent_flag)):
        agent_i = np.array([exp[:,i] for exp in result_list])
        new_list.append(np.mean(agent_i,axis=0))
        new_list1.append(np.std(agent_i,axis=0))
    strategic_variables = np.array(new_list).T
    strategic_variables = strategic_variables[memory_step:,:] if not show_memory else strategic_variables
    strategic_variables = strategic_variables[:-test_step,:] if not show_test else strategic_variables
    strategic_variables_std = np.array(new_list1).T
    x = pd.DataFrame(strategic_variables_std.mean(axis=1).reshape(-1,24).mean(axis=1))
    x.to_csv(os.path.join(result_path,'average_episodic_strategic_std.csv'),index=False)
    strategic_variables_std = strategic_variables_std[memory_step:,:] if not show_memory else strategic_variables_std
    strategic_variables_std = strategic_variables_std[:-test_step,:] if not show_test else strategic_variables_std




    ##-------------------------------决策系数gamma---------------------------------
    if action_gamma:
        # 先save test strategic
        strategic_variables_gamma = pd.read_csv(os.path.join(result_path, 'result_action_beta.csv'))
        strategic_variables_gamma = strategic_variables_gamma.values
        # 取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, strategic_variables_gamma.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = strategic_variables_gamma[i:i + n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)
        new_list = []
        new_list1 = []
        for i in range(len(agent_flag)):
            agent_i = np.array([exp[:, i] for exp in result_list])
            new_list.append(np.mean(agent_i, axis=0))
            new_list1.append(np.std(agent_i, axis=0))

        strategic_all = np.array(new_list).T
        x1 = strategic_all[-test_step:, :].mean(0)
        x2 = strategic_all[-test_step:, :].std(0)  # 主要用来看看这个是不是真收敛了
        # x = pd.DataFrame((x1, x2))
        ## x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        # x.to_csv(os.path.join(result_path, 'test_strategic_gamma_all.csv'), index=False, header=None)

        strategic_all = np.array(new_list1).T
        x3 = strategic_all[-test_step:, :].mean(0) #不同实验的std
        # x2 = strategic_all[-test_step:, :].std(0)  # 主要用来看看这个是不是真收敛了
        x = pd.DataFrame((x1, x2, x3))
        # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #计算所有参与者20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        x.to_csv(os.path.join(result_path, 'test_strategic_gamma_all.csv'), index=False, header=None)



        # 开始
        strategic_variables_gamma = pd.read_csv(os.path.join(result_path, 'result_action_beta.csv'))
        # strategic_variables_gamma = strategic_variables_gamma.values * agent_flag
        # is_zero_column = np.all(strategic_variables_gamma == 0, axis=0)
        # non_zero_columns = np.where(~is_zero_column)[0]
        strategic_variables_gamma = strategic_variables_gamma.values[:, non_zero_columns]
        strategic_variables_gamma1 = strategic_variables_gamma.copy()

        # 取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, strategic_variables_gamma.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = strategic_variables_gamma[i:i + n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)

        new_list = []
        new_list1 = []
        for i in range(sum(agent_flag)):
            agent_i = np.array([exp[:, i] for exp in result_list])
            new_list.append(np.mean(agent_i, axis=0))
            new_list1.append(np.std(agent_i, axis=0))
        strategic_variables_gamma = np.array(new_list).T
        strategic_variables_gamma = strategic_variables_gamma[memory_step:, :] if not show_memory else strategic_variables_gamma
        strategic_variables_gamma = strategic_variables_gamma[:-test_step, :] if not show_test else strategic_variables_gamma
        strategic_variables_gamma_std = np.array(new_list1).T
        x = pd.DataFrame(strategic_variables_gamma_std.mean(axis=1).reshape(-1, 24).mean(axis=1))
        x.to_csv(os.path.join(result_path, 'average_episodic_strategic_gamma_std.csv'), index=False)
        strategic_variables_gamma_std = strategic_variables_gamma_std[memory_step:, :] if not show_memory else strategic_variables_gamma_std
        strategic_variables_gamma_std = strategic_variables_gamma_std[:-test_step, :] if not show_test else strategic_variables_gamma_std


    # --------------------------reward--------------------------------------
    # 先save test reward
    rewards = pd.read_csv(os.path.join(result_path,'result_reward.csv'))
    rewards = rewards.values
    #取出每个实验的数据，存入列表
    result_list = []
    # 遍历矩阵的行
    for i in range(0, rewards.shape[0], n_steps_experiment):
        # 截取每个子列表的行范围
        chunk = rewards[i:i+n_steps_experiment]
        # 将子列表添加到结果列表中
        result_list.append(chunk)
    new_list = []
    new_list1 = []
    for i in range(len(agent_flag)):
        agent_i = np.array([exp[:,i] for exp in result_list])
        new_list.append(np.mean(agent_i,axis=0))
        new_list1.append(np.std(agent_i,axis=0))
    rewards_all = np.array(new_list).T
    # x1 = rewards_all[-test_step:,:].sum(0) #test day这么多天的总共profit
    x1 = rewards_all[-test_step:,:].mean(0) #test day这么多天的平均profit （建议，容易与std同量纲意义对比数据）
    x2 = rewards_all[-test_step:,:].std(0) #主要用来看看这个是不是真收敛了
    # x = pd.DataFrame((x1,x2))
    # x = pd.DataFrame(strategic_all[-test_step:,:].mean(0)) #得到所有agent在test阶段中所有回合的总利润
    # x.to_csv(os.path.join(result_path,'test_reward_all.csv'),index=False,header=None)
    rewards_all = np.array(new_list1).T
    x3 = rewards_all[-test_step:, :].mean(0)  # 不同实验的std
    x = pd.DataFrame((x1, x2, x3))
    x.to_csv(os.path.join(result_path, 'test_reward_all.csv'), index=False, header=None)

    #开始
    rewards = pd.read_csv(os.path.join(result_path,'result_reward.csv'))
    # rewards = rewards.values * agent_flag
    # is_zero_column = np.all(rewards == 0, axis=0)
    # non_zero_columns = np.where(~is_zero_column)[0]
    # non_zero_columns = np.where(np.array(agent_flag) == 1)[0]
    rewards = rewards.values[:, non_zero_columns]
    rewards1 = rewards.copy()

    #取出每个实验的数据，存入列表
    result_list = []
    # 遍历矩阵的行
    for i in range(0, rewards.shape[0], n_steps_experiment):
        # 截取每个子列表的行范围
        chunk = rewards[i:i+n_steps_experiment]
        # 将子列表添加到结果列表中
        result_list.append(chunk)

    new_list = []
    new_list1 = []
    for i in range(sum(agent_flag)):
        agent_i = np.array([exp[:,i] for exp in result_list])
        new_list.append(np.mean(agent_i,axis=0))
        new_list1.append(np.std(agent_i, axis=0))
    rewards = np.array(new_list).T
    rewards = rewards[memory_step:, :] if not show_memory else rewards
    rewards = rewards[:-test_step, :] if not show_test else rewards
    rewards_std = np.array(new_list1).T
    rewards_std = rewards_std[memory_step:, :] if not show_memory else rewards_std
    rewards_std = rewards_std[:-test_step, :] if not show_test else rewards_std


    # --------------------------reward对比--------------------------------------
    if algo != 'Q-learning' and algo != 'DQN':
        rewards1 = pd.read_csv(os.path.join(result_path,'reward_agent_nonagent_mean.csv'))
        rewards1 = rewards1.values
        # is_zero_column = np.all(rewards1 == 0, axis=0)
        # non_zero_columns = np.where(~is_zero_column)[0]
        # rewards1 = rewards1[:, non_zero_columns]

        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, rewards1.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = rewards1[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)

        new_list = []
        new_list1 = []
        for i in range(rewards1.shape[1]):
            agent_i = np.array([exp[:,i] for exp in result_list])
            new_list.append(np.mean(agent_i,axis=0))
            new_list1.append(np.std(agent_i,axis=0))
        rewards1 = np.array(new_list).T
        rewards1 = rewards1[memory_step:, :] if not show_memory else rewards1
        rewards1 = rewards1[:-test_step, :] if not show_test else rewards1
        rewards1_std = np.array(new_list1).T
        rewards1_std = rewards1_std[memory_step:, :] if not show_memory else rewards1_std
        rewards1_std = rewards1_std[:-test_step, :] if not show_test else rewards1_std

        # --------------------------price对比--------------------------------------
        #保存一下原始price
        price = pd.read_csv(os.path.join(result_path,'result_price.csv'))
        price = price.values
        # is_zero_column = np.all(price == 0, axis=0)
        # non_zero_columns = np.where(~is_zero_column)[0]
        # price = price[:, non_zero_columns]
        price1 = price.copy()
        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, price.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = price[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)
        new_list = []
        for i in range(len(agent_flag)):
            agent_i = np.array([exp[:, i] for exp in result_list])
            new_list.append(np.mean(agent_i, axis=0))
        price_all = np.array(new_list).T
        x = pd.DataFrame(price_all[-test_step:, :].mean(0))  # 计算20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        x.to_csv(os.path.join(result_path, 'test_price_all.csv'), index=False, header=None)


        price = pd.read_csv(os.path.join(result_path,'price_agent_nonagent_mean.csv'))
        price = price.values
        # is_zero_column = np.all(price == 0, axis=0)
        # non_zero_columns = np.where(~is_zero_column)[0]
        # price = price[:, non_zero_columns]

        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, price.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = price[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)

        new_list = []
        new_list1 = []
        for i in range(rewards1.shape[1]):
            agent_i = np.array([exp[:,i] for exp in result_list])
            new_list.append(np.mean(agent_i,axis=0))
            new_list1.append(np.std(agent_i,axis=0))
        price = np.array(new_list).T
        price = price[memory_step:, :] if not show_memory else price
        price = price[:-test_step, :] if not show_test else price
        price_std = np.array(new_list1).T
        price_std = price_std[memory_step:, :] if not show_memory else price_std
        price_std = price_std[:-test_step, :] if not show_test else price_std


        # --------------------------quantity对比--------------------------------------
        #保存一下原始quantity
        quantity = pd.read_csv(os.path.join(result_path,'result_quantity.csv'))
        quantity = quantity.values
        # is_zero_column = np.all(quantity == 0, axis=0) #以下三句有问题，直接注释掉
        # non_zero_columns = np.where(~is_zero_column)[0]
        # quantity = quantity[:, non_zero_columns]
        quantity1 = quantity.copy()

        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, quantity.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = quantity[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)
        new_list = []
        for i in range(len(agent_flag)):
            agent_i = np.array([exp[:, i] for exp in result_list])
            new_list.append(np.mean(agent_i, axis=0))
        quantity_all = np.array(new_list).T
        x = pd.DataFrame(quantity_all[-test_step:, :].mean(0))  # 计算20个回合的策略收敛平均值(直接取其中一个都行，一般都是相等的)
        x.to_csv(os.path.join(result_path, 'test_quantity_all.csv'), index=False, header=None)


        # ---------开始对比
        quantity = pd.read_csv(os.path.join(result_path,'quantity_agent_nonagent_mean.csv'))
        quantity = quantity.values
        # is_zero_column = np.all(quantity == 0, axis=0)
        # non_zero_columns = np.where(~is_zero_column)[0]
        # quantity = quantity[:, non_zero_columns]

        #取出每个实验的数据，存入列表
        result_list = []
        # 遍历矩阵的行
        for i in range(0, quantity.shape[0], n_steps_experiment):
            # 截取每个子列表的行范围
            chunk = quantity[i:i+n_steps_experiment]
            # 将子列表添加到结果列表中
            result_list.append(chunk)

        new_list = []
        new_list1 = []
        for i in range(rewards1.shape[1]):
            agent_i = np.array([exp[:,i] for exp in result_list])
            new_list.append(np.mean(agent_i,axis=0))
            new_list1.append(np.std(agent_i,axis=0))
        quantity = np.array(new_list).T
        quantity = quantity[memory_step:, :] if not show_memory else quantity
        quantity = quantity[:-test_step, :] if not show_test else quantity
        quantity_std = np.array(new_list1).T
        quantity_std = quantity_std[memory_step:, :] if not show_memory else quantity_std
        quantity_std = quantity_std[:-test_step, :] if not show_test else quantity_std

    # --------------------------average_episodic_rewards--------------------------------------
    episodic_rewards = pd.read_csv(os.path.join(result_path,'average_episodic_rewards.csv'))
    episodic_rewards = episodic_rewards.values
    episodic_rewards = episodic_rewards[memory_episode:, :] if not show_memory else episodic_rewards
    episodic_rewards = episodic_rewards[:-int(test_step/episode_length), :] if not show_test else episodic_rewards
    episodic_rewards_std = pd.read_csv(os.path.join(result_path,'average_episodic_rewards_std.csv'))
    episodic_rewards_std = episodic_rewards_std.values
    episodic_rewards_std = episodic_rewards_std[memory_episode:, :] if not show_memory else episodic_rewards_std
    episodic_rewards_std = episodic_rewards_std[:-int(test_step/episode_length), :] if not show_test else episodic_rewards_std


    if time_vary_data_need:
        episodic_rewards_test = pd.read_csv(os.path.join(result_path,'test_average_episodic_rewards.csv'))
        episodic_rewards_test = episodic_rewards_test.values
        episodic_rewards_test_std = pd.read_csv(os.path.join(result_path,'test_average_episodic_rewards_std.csv'))
        episodic_rewards_test_std = episodic_rewards_test_std.values





    # -----------------------------画图---------------------------
    n_agents = 8 if case_name == 'case4_disp' else 9 if case_name == 'case30_disp_self' else 10
    n_agents_gen = 4 if case_name == 'case4_disp' else 6 if case_name == 'case30_disp_self' else 7
    label1 = ["gen {}".format(i + 1) for i in range(n_agents_gen)]
    label2 = ["load {}".format(i + 1) for i in range(n_agents - n_agents_gen)]
    label3 = label1 + label2
    # color = ['#E94849', '#5D93BF', '#71BE6E', '#FF8000']  # 红蓝绿橙，浅
    # color = ['#FFB3B3', '#97D4F2', '#AFD99B', '#FFEA80'] #红蓝绿黄，更浅，另一种黄：#F4E478
    # color = ['#FF0000', '#0000FF', '#00FF00', '#828282']  # 红蓝绿灰，更深 #FF8000  #939294
    color = ['#FF0000', '#0000FF', '#858585', '#00FF00']  # 红蓝绿灰，更深 #FF8000
    # color = ['#F8CBAD', '#B4C7E7', '#C5E0B4', '#F2F2F2']  # 红蓝绿橙，更浅
    color = ['#FF0000', '#0000FF', '#858585', '#00FF00', '#800000', '#FF8000']  # 加个棕色
    # color2 = ['#FFC0CB', '#00FFFF', '#800080']  #粉 蓝绿 紫红
    color2 = color[:3]
    # color2 = ['#FFC0CB', '#00FFFF', '#800080']  #

    agent_id = np.where(np.array(agent_flag) == 1)[0]
    agent_label = list(np.array(label3)[agent_id])  # 外面套不套lis都行, (x,)的ndarray也可以
    # Visualize the data
    clip_a = 0.5
    clip_b = 0.5

    if sum(agent_flag) > 6:
        color=None
    # color=None

    # --------------------------------------------------------------
    # 对数组进行卷积操作并进行零填充
    # mean_kernel_len_epi = 20
    # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
    # for i in range(strategic_variables.shape[-1]):
    #     strategic_variables[:, i] = np.convolve(strategic_variables[:, i], kernel_epi, mode='same')
    # strategic_variables = strategic_variables[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    # for i in range(strategic_variables.shape[-1]):
    #     strategic_variables_std[:, i] = np.convolve(strategic_variables_std[:, i], kernel_epi, mode='same')
    # strategic_variables_std = strategic_variables_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    smooth_rate = smooth_rate_default
    strategic_variables = smooth(strategic_variables, smooth_rate) #只画训练集
    strategic_variables_std = smooth(strategic_variables_std, smooth_rate) #只画训练集
    visualize_sequence_data(strategic_variables[:,:], agent_label, color=color,
                            xlabel='Iterations (Trading Periods)', ylabel=r'strategic coefficient: $\alpha$',
                            filename=os.path.join(result_path,'episodic_average_agent_action.png'), test_step=test_step, memory_step=memory_step, fill_std=strategic_variables_std)



    if action_gamma:
        # 对数组进行卷积操作并进行零填充
        # mean_kernel_len_epi = 20
        # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
        # for i in range(strategic_variables_gamma.shape[-1]):
        #     strategic_variables_gamma[:, i] = np.convolve(strategic_variables_gamma[:, i], kernel_epi, mode='same')
        # strategic_variables_gamma = strategic_variables_gamma[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        # for i in range(strategic_variables_gamma.shape[-1]):
        #     strategic_variables_gamma_std[:, i] = np.convolve(strategic_variables_gamma_std[:, i], kernel_epi, mode='same')
        # strategic_variables_gamma_std = strategic_variables_gamma_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        smooth_rate = smooth_rate_default
        strategic_variables_gamma = smooth(strategic_variables_gamma, smooth_rate)  # 只画训练集
        strategic_variables_gamma_std = smooth(strategic_variables_gamma_std, smooth_rate)  # 只画训练集
        visualize_sequence_data(strategic_variables_gamma[:,:], agent_label, color=color,
                                xlabel='Iterations (Trading Periods)', ylabel=r'$\beta$',
                                filename=os.path.join(result_path,'episodic_average_agent_action_beta.png'), test_step=test_step, memory_step=memory_step, fill_std=strategic_variables_gamma_std,std_scale=0.1)



    # 对数组进行卷积操作并进行零填充
    # mean_kernel_len_epi = 30
    # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
    # for i in range(rewards.shape[-1]):
    #     rewards[:, i] = np.convolve(rewards[:, i], kernel_epi, mode='same')
    # rewards = rewards[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    # for i in range(rewards.shape[-1]):
    #     rewards_std[:, i] = np.convolve(rewards_std[:, i], kernel_epi, mode='same')
    # rewards_std = rewards_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    smooth_rate = smooth_rate_default
    rewards = smooth(rewards, smooth_rate)  # 只画训练集
    rewards_std = smooth(rewards_std, smooth_rate)  # 只画训练集
    visualize_sequence_data(rewards[:,:], agent_label, color=color, #memory_step:
                            xlabel='Iterations (Trading Periods)', ylabel=r'Reward($10^3\$$)',
                            filename=os.path.join(result_path,'episodic_average_agent_reward.png'), test_step=test_step, memory_step=memory_step, fill_std=rewards_std, std_scale=0.3)


    #综合以上两个图
    # plt.figure(figsize=(6, 14))
    # plt.figure(figsize=(10,12))
    plt.figure(figsize=(6,12))
    # plt.subplot(3,1,1)
    plt.subplot(2,1,1)
    visualize_sequence_data(strategic_variables[:,:], agent_label, color=color,
                            # xlabel='Iterations (Trading Periods)\n(a)', ylabel=r'strategic coefficient: $\alpha$',
                            xlabel='Iterations (Trading Periods)\n(c)', ylabel=r'strategic coefficient: $\alpha$',
                            filename=os.path.join(result_path,'episodic_average_agent_action.png'), test_step=test_step, memory_step=memory_step, fill_std=strategic_variables_std, subplot=True)
    # plt.subplot(3,1,2)
    plt.subplot(2,1,2)
    visualize_sequence_data(rewards[:,:], agent_label, color=color, #memory_step:
                            # xlabel='Iterations (Trading Periods)\n(b)', ylabel=r'Reward($10^3\$$)',
                            xlabel='Iterations (Trading Periods)\n(d)', ylabel=r'Reward($10^3\$$)',
                            filename=os.path.join(result_path,'episodic_average_agent_reward.png'), test_step=test_step, memory_step=memory_step, fill_std=rewards_std, std_scale=0.3, subplot=True, show_legend=True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path,'subplots1.png'), dpi=200)


    # -------------------------------------------------------------------
    # 对数组进行卷积操作并进行零填充
    if algo != 'Q-learning' and algo != 'DQN':
        # mean_kernel_len_epi = 20
        # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
        # for i in range(rewards1.shape[-1]):
        #     rewards1[:, i] = np.convolve(rewards1[:, i], kernel_epi, mode='same')
        # rewards1 = rewards1[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        # for i in range(rewards1.shape[-1]):
        #     rewards1_std[:, i] = np.convolve(rewards1_std[:, i], kernel_epi, mode='same')
        # rewards1_std = rewards1_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        smooth_rate = smooth_rate_default
        rewards1 = smooth(rewards1, smooth_rate)  # 只画训练集
        rewards1_std = smooth(rewards1_std, smooth_rate)  # 只画训练集
        visualize_sequence_data(rewards1[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color,
                                xlabel='Iterations (Trading Periods)', ylabel=r'Average Reward($ 10^3\$ $)',
                                filename=os.path.join(result_path,'episodic_average_agent_reward1.png'), test_step=test_step, memory_step=memory_step, fill_std=rewards1_std)


        # 对数组进行卷积操作并进行零填充
        # mean_kernel_len_epi = 20
        # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
        # for i in range(price.shape[-1]):
        #     price[:, i] = np.convolve(price[:, i], kernel_epi, mode='same')
        # price = price[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        # for i in range(price.shape[-1]):
        #     price_std[:, i] = np.convolve(price_std[:, i], kernel_epi, mode='same')
        # price_std = price_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        smooth_rate = smooth_rate_default
        price = smooth(price, smooth_rate)  # 只画训练集
        price_std = smooth(price_std, smooth_rate)  # 只画训练集
        visualize_sequence_data(price[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color,
                                xlabel='Iterations (Trading Periods)', ylabel=r'Price($/MWh)',
                                filename=os.path.join(result_path,'episodic_average_agent_price.png'), test_step=test_step, memory_step=memory_step, fill_std=price_std)


        # 对数组进行卷积操作并进行零填充
        # mean_kernel_len_epi = 20
        # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
        # for i in range(quantity.shape[-1]):
        #     quantity[:, i] = np.convolve(quantity[:, i], kernel_epi, mode='same')
        # quantity = quantity[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        # for i in range(quantity.shape[-1]):
        #     quantity_std[:, i] = np.convolve(quantity_std[:, i], kernel_epi, mode='same')
        # quantity_std = quantity_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
        smooth_rate = smooth_rate_default
        quantity = smooth(quantity, smooth_rate)  # 只画训练集
        quantity_std = smooth(quantity_std, smooth_rate)  # 只画训练集
        visualize_sequence_data(quantity[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color,
                                xlabel='Iterations (Trading Periods)', ylabel=r'Quantity(MWh)',
                                filename=os.path.join(result_path,'episodic_average_agent_quantity.png'), test_step=test_step, memory_step=memory_step, fill_std=quantity_std)


        #综合以上三个图
        # plt.figure(figsize=(6, 14))
        plt.figure(figsize=(18, 6))
        # plt.subplot(3,1,1)
        plt.subplot(1,3,1)
        visualize_sequence_data(rewards1[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color2,
                                # xlabel='Iterations (Trading Periods)\n(a)', ylabel=r'Average Reward($ 10^3\$ $)',
                                xlabel='Iterations (Trading Periods)\n(d)', ylabel=r'Average Reward($ 10^3\$ $)',
                                filename=os.path.join(result_path,'episodic_average_agent_reward1.png'), test_step=test_step, memory_step=memory_step, subplot=True, fill_std=rewards1_std)
        # plt.subplot(3,1,2)
        plt.subplot(1,3,2)
        visualize_sequence_data(price[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color2,
                                # xlabel='Iterations (Trading Periods)\n(b)', ylabel=r'Price($/MWh)',
                                xlabel='Iterations (Trading Periods)\n(e)', ylabel=r'Price($/MWh)',
                                filename=os.path.join(result_path,'episodic_average_agent_price.png'), test_step=test_step, memory_step=memory_step, subplot=True, fill_std = price_std, show_legend=True)
        # plt.legend(loc='lower right')
        # plt.subplot(3,1,3)
        plt.subplot(1,3,3)
        visualize_sequence_data(quantity[:,:], labels=['strategic agents', 'non-strategic agents', 'total'], color=color2,
                                # xlabel='Iterations (Trading Periods)\n(c)', ylabel=r'Quantity(MWh)',
                                xlabel='Iterations (Trading Periods)\n(f)', ylabel=r'Quantity(MWh)',
                                filename=os.path.join(result_path,'episodic_average_agent_quantity.png'), test_step=test_step, memory_step=memory_step, subplot=True, fill_std = quantity_std, show_legend=True)
        # axes = plt.gcf().get_axes()
        # for ax in axes:
        #     # ax.set_position([0.1, 0.1, 0.4, 0.4])
        #
        # fig = plt.gcf() #创建共用的图例
        # handles, labels = plt.gca().get_legend_handles_labels()
        # fig.legend(handles, labels,loc='center')
        # plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path,'subplots2.png'), dpi=200)


    # mean_kernel_len_epi = 10
    # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
    # for i in range(quantity.shape[-1]):
    #     quantity[:, i] = np.convolve(quantity[:, i], kernel_epi, mode='same')
    # quantity = quantity[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    # for i in range(quantity.shape[-1]):
    #     quantity_std[:, i] = np.convolve(quantity_std[:, i], kernel_epi, mode='same')
    # quantity_std = quantity_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b), :]
    # mean_kernel_len_epi = 4
    # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
    # episodic_rewards = np.convolve(np.array(episodic_rewards).squeeze(), kernel_epi, mode='same')
    # episodic_rewards = episodic_rewards[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b)]
    # episodic_rewards_std = np.convolve(np.array(episodic_rewards_std).squeeze(), kernel_epi, mode='same')
    # episodic_rewards_std = episodic_rewards_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b)]
    smooth_rate = 0.3
    episodic_rewards = smooth(episodic_rewards, smooth_rate)  # 只画训练集
    episodic_rewards_std = smooth(episodic_rewards_std, smooth_rate)  # 只画训练集
    visualize_sequence_data(episodic_rewards.reshape(-1,1), color=color,
                            xlabel='Episode', ylabel=r'Average Reward($10^3\$$)',
                            filename=os.path.join(result_path,'average_episodic_rewards.png'), test_step=int(test_step/24), memory_step=memory_episode, fill_std=episodic_rewards_std.reshape(-1, 1))


    if time_vary_data_need:
        # mean_kernel_len_epi = 2
        # kernel_epi = np.ones(mean_kernel_len_epi) / mean_kernel_len_epi
        # episodic_rewards = np.convolve(np.array(episodic_rewards).squeeze(), kernel_epi, mode='same')
        # episodic_rewards = episodic_rewards[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b)]
        # episodic_rewards_std = np.convolve(np.array(episodic_rewards_std).squeeze(), kernel_epi, mode='same')
        # episodic_rewards_std = episodic_rewards_std[int(mean_kernel_len_epi * clip_a):-int(mean_kernel_len_epi * clip_b)]
        visualize_sequence_data(episodic_rewards_test.reshape(-1,1), color=color,
                                xlabel='Episode', ylabel=r'Average Reward($10^3\$$)',
                                filename=os.path.join(result_path,'test_average_episodic_rewards.png'), test_step=None, memory_step=None, fill_std=episodic_rewards_test_std.reshape(-1, 1))




if __name__ == '__main__':
    result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case57_disp_self\congested\DQN\[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\10-11_17-00-58_GCN'
    result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case4_disp\congested\MADDPG\[1, 0, 0, 1, 1, 1, 1, 1]\12-14_16-14-19_gamma_0.9'
    result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case30_disp_self\congested\MADDPG\[1, 0, 1, 1, 0, 0, 1, 0, 0]\12-20_17-09-57_gamma_0.9_ok'
    result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case4_disp\congested\MADDPG\[1, 0, 1, 1, 0, 1, 0, 0]\12-21_08-54-42_gamma_0.9_ok'

    # result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case4_disp\congested\MADDPG\[1, 0, 1, 1, 0, 1, 0, 0]\01-08_10-17-50_gamma_0.9_ok'
    result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case4_disp\congested\MADDPG\[1, 0, 1, 0, 1, 1, 1, 1]_ok\01-05_19-42-08_gamma_0.9'
    # result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case30_disp_self\congested\MADDPG\[1, 0, 1, 1, 0, 0, 1, 0, 0]\12-20_17-09-57_gamma_0.9_ok'
    # result_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case30_disp_self\congested\MADDPG\[0, 0, 0, 1, 0, 0, 1, 1, 1]_ok\01-04_10-46-56_gamma_0.9_ok'
    algo = 'MADDPG' #MADDPG DDPG Q-learning MADDPG-PER DQN
    case_name = 'case4_disp' #case30_disp_self or case4_disp or case57_disp_self case30需要更多轮训练
    time_vary_data_need = False

    # agent_flag = [1, 0, 1, 1,  0, 1, 0, 0]
    agent_flag = [1, 0, 1, 0, 1, 1, 1, 1]
    # agent_flag = [1, 0, 1, 1, 0, 0,   1, 0, 0]
    # agent_flag = [0, 0, 0, 1, 0, 0, 1, 1, 1]
    episode_num = 220  # 220 170 for gamma test
    memory_step = 1000

    plot_experiment(result_path, algo, case_name, time_vary_data_need, agent_flag, episode_num, memory_step, 0.96)