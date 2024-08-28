import argparse
import os, sys
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径

import numpy as np
from tqdm import tqdm
from market.multi_choice_bus import market_clearing
from algorithm.MADDPG_beifen_vwork2 import MADDPG, MADDPG_true, MADDPG_GCN
from algorithm.model_beifen_vwork2 import * #实测2 3没什么区别
from plot_experiment_plus import plot_experiment
import matlab.engine

import pandas as pd
import tensorflow as tf
from utils import *
from datetime import datetime
import torch
import random



np.set_printoptions(precision=4)
torch.set_printoptions(precision=2)
#如何在开启tensorboard同时清除之前窗口：
# 首先删掉log文件夹所有数据
# 终端输入代码：tensorboard --logdir=D:\pycharmproject3\RLcode\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\log --host=localhost --port=6006 --purge_orphaned_data=true

# 不删除旧数据直接启动：
# tensorboard --logdir=D:\pycharmproject3\RLcode\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\log
# 注意不可以：
# tensorboard --logdir=r'D:\pycharmproject3\RLcode\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\log'


##-------------------------------一：各种超参数-------------------------------
show_tensorb = False #节省很多显存
if show_tensorb:
    log_dir = r'D:\pycharmproject3\RLcode\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\log'
    writer = tf.summary.create_file_writer(log_dir) # 创建一个用于记录日志的SummaryWriter

gen_low = 0.90
gen_high = 2.0
load_low = 0.85
load_high = 1.0

quantity_high = 1
quantity_low = 0.80

gamma_default = 0.90
# gamma_default = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 1] * 3
auto_gamma_low = 0
auto_gamma_high = gamma_default
auto_gamma_decay = 0.98 #越小，gama升的越快
#auto gamma test example
# x = auto_gamma_low
# x_list = []
# for i in range(300): #300 step
#     x = auto_gamma_high - auto_gamma_decay * (auto_gamma_high - x)
#     x_list.append(x)

lr_default = 0.0003#0.001   0.0003  #
# lr_default = [0.01, 0.005, 0.001, 0.0005] * 5 #0.001   0.0003  #
# lr_default = [0.005, 0.001, 0.0005] * 3 #0.001   0.0003  #
auto_lr_low = lr_default
auto_lr_high = lr_default * 2 #3 5没什么差别，2好点好像
auto_lr_decay = 0.99

soft_rate = 0.01 #default 0.01
# soft_rate = [0.1, 0.01, 0.001] * 3 #default 0.01

sed = [42,0,1,2,5,10,11,99,88,77, 35719,1000,20,55,12321,2354,123,425,467,1276] #4-bus 30 bus
# sed = [42,3407,30519,88,66,10,11,99,88,77, 35719,1000,20,55,12321,2354,123,425,467,1276]
# sed = [11,99,88,77, 35719,1000,20,55,12321,2354,123,425,467,1276]
# sed = [42,42,42,42,42,42,42,42,42]



# sed = [[i]*5 for i in sed] #这两句用于用多个训练好的模型去推理多次
# sed = [item for sublist in sed for item in sublist]

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
train_or_test = 'train' # train or test
# test_save_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case30_disp_self\congested\DDPG\[1, 1, 1, 1, 0, 0, 0, 1, 1]\10-09_09-16_gamma_0.9_load_data_GCN_ada_ok'
# test_save_path = r'E:\pycharm_project\agent-based-modeling-in-electricity-market-using-DDPG-algorithm-master\results\case30_disp_self\congested\DDPG\[1, 1, 1, 1, 0, 0, 0, 1, 1]\10-08_17-23_gamma_0.9_load_data_ok'

test_save_path = test_save_path + '/'
# test_model_ex_dir = [0,1,2,3,4] * 20 #[2] #list长度多少，就test不同模型多少次，取平均结果，list中每个元素，表示每轮test取第几个不同的模型，下行同理
test_model_ex_dir = [0]  #[2] #list长度多少，就test不同模型多少次，取平均结果，list中每个元素，表示每轮test取第几个不同的模型，下行同理
# test_round = [5279] *5 *20 #5040 5280
test_round = [3960]  #5040 5280       6840 0      6840 3 6840 3 6480 2
# test_round = 4320 #5040 5280

case_name = 'case57_disp_self' #case30_disp_self or case4_disp or case57_disp_self
congest_case = 'congested' #uncongested
alg_name = 'DDPG'
GCN = True
MADDPG_critic_input = 1 #0表示critic输入只考虑其他agent的动作和状态，1表示考虑所有其他agent和非agent的动作和状态，默认为1
pre_action = False #是否将前一时刻的动作作为状态之一,实测没什么用
quantity_action = False
auto_gamma = False #递增的gamma
auto_lr = False  #递减的lr
max_load_required = False #是否将最大负荷需求作为状态之一 ，若是固定值，实测对训练起副作用，若是变化值，好像有用的，但是不用好像也可以，没有变化太大
load_data_need = False
full_knowledge = True #只能用于DDPG，但是目测没写好，MADDPG没用，不需要
# agent_flag = [1,0,1,0, 1,1,1,1] if case_name == 'case4_disp' else [0,0,0,0,0,0, 1,1,1]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
# agent_flag = [1,0,0,0, 0,1,0,0] if case_name == 'case4_disp' else [1,1,0,0,0,0, 0,1,1]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
# agent_flag = [1,0,1,1, 0,1,0,0] if case_name == 'case4_disp' else [1,0,1,1,0,0, 1,0,0]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
# agent_flag = [0,1,0,0, 1,0,1,1] if case_name == 'case4_disp' else [1,0,1,1,0,0, 1,0,0]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
# agent_flag = [1,1,1,1, 1,1,1,1] if case_name == 'case4_disp' else [1,1,1,0,0,0, 1,1,1] if case_name == 'case30_disp_self' else [1,1,1,1,1,0,0]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
agent_flag = [1,1,0,0, 0,0,1,1] if case_name == 'case4_disp' else [1,1,1,1,0,0, 0,1,1] if case_name == 'case30_disp_self' else [1,1,1,1,1,1,1, 1,1,1] #[1,1,0,1,0,1,1, 0,1,1]
# agent_flag = [1,1,1,1, 0,0,0,0] if case_name == 'case4_disp' else [0,0,0,1,1,1, 1,0,1]  #[1,0,1,1, 0,0,0,0]这一组可以突出base（全1）、ddpg、maddpg之间的差别，如果是[x,x,1,x,0,0,0,0]就是one side
# agent_flag = [1,1,1,1, 1,1,1,1] if case_name == 'case4_disp' else [1,1,1,1,1,1, 1,1,1]
# agent_flag = [0,0,0,0, 1,1,1,1] if case_name == 'case4_disp' else [1,1,1,1,1,1, 0,0,0]
# quantity_flag = agent_flag if quantity_action else None
quantity_flag = [0,0,0,0, 1,1,1,1] if quantity_action else [0]*len(agent_flag)
agent_true_id = [id for id,i in enumerate(agent_flag) if i == 1]
n_agents_true = len(agent_true_id) #真正是agent的有几个

agent_id = np.where(np.array(agent_flag) == 1)
nonagent_id = np.where(np.array(agent_flag) == 0)

experiment_times = 6 if train_or_test == 'train' else len(test_model_ex_dir)
memory_capacity = 1000 #效果不好就加大这个，训练轮数反而不用太多也可以
episode_length = 24 #一局走多少步，如果没有局的概念，可以设为None
test_episode = 20 #30  #2
test_step = int(24*test_episode) #设为0则在最后不进行测试
evaluate_rate_episode = 15
evaluate_begin_step = 2000 # =memory_capacity
train_times = 6.0
train_day_begin_idx = 10226  #0 8762,第二年5 1   746 6月
test_day_begin_idx = 10970 #0 10226 7月   14643 23 1月  14500 2.5%  10970
load_data_train_episode = 200 #需要用的数据的天数长度   123,对应5-8月    61   425    273 6-2月数据     200   30
episode_num = int(load_data_train_episode*train_times+test_episode) if load_data_need else load_data_train_episode+test_episode #220    170 for gamma test
# n_steps = 8000 if not episode_length else episode_length*episode_num
n_steps = int(episode_length*episode_num)
n_agents = 8 if case_name == 'case4_disp' else 9 if case_name == 'case30_disp_self' else 10
n_agents_gen = 4 if case_name == 'case4_disp' else 6 if case_name == 'case30_disp_self' else 7
n_actions = 2 if not quantity_action else 2 #2=price quantity pair bidding
# action_id = {'price':0, 'quantity':1}


#用于做line outage实验
branch_disc = False
bus_disc = False
gen_disc = False
dis_num = 2 #1 3 5 8 10 15


#early stopping以及探索参数设置
training_stop_judge = 'rewards_agent_mean' #‘action’ or 'rewards_agent_mean' 是根据agent的action是否收敛来判定程序停止,还是agent的reward总和
training_std_end = 0.005 #当达到此数，训练终止,注意不同training_stop_judge，设置可能不同，用action可能更通用一些
training_std_length = 200 #标准差计算的步数长度
training_step_least = 1000 #最少训练的步数
var = 0.5 # 上下限也就[-1,1]，没必要过大  0.5
var_low = 0.03 #0.03
var_low_step = 24*10 #保持在var_low这个var训练的步数   10


# test_learn_step = 500 #相当于贪心训练的时间步数
# test_learn_var = 0.01 #贪心训练，但保留微小var
adaptive_var_decay_rate = True

if adaptive_var_decay_rate:
    # 衰减方程：var * var_decay_rate**(n_steps - memory_capacity - var_low_step - test_step) == var_low
    var_decay_rate = np.exp(np.log(var_low/var) / (n_steps - memory_capacity - var_low_step - test_step)) #减去memory_capacity，是因为var的衰减是从网络开始更新之后(memory填满)才开始衰减的
    print("var_decay_rate:{:5}".format(var_decay_rate))
    #var衰减的步数：n_steps - memory_capacity - var_low_step - test_step
    #训练的步数：n_steps - memory_capacity - test_step
else:
    var_decay_rate = 0.9993

##save
now = datetime.now()  # 获取当前时间
current_time = now.strftime("%m-%d_%H-%M-%S")  # 将当前时间格式化成字符串
# current_time = current_time + '_gamma_' + str(gamma_default)
current_time = current_time + '_gamma_' + str(gamma_default[:3] if type(gamma_default) == list else gamma_default)
# current_time = current_time + '_lr_' + str(lr_default)
# current_time = current_time + '_lr_' + str(lr_default[:3] if type(lr_default) == list else lr_default)
# current_time = current_time + '_tau_' + str(soft_rate)
# current_time = current_time + '_tau_' + str(soft_rate[:3] if type(soft_rate) == list else soft_rate)
#在 PyCharm 运行脚本时，默认的当前工作目录是该脚本所在的文件夹的目录，而不是整个工程项目的根目录。
save_path = r'../results/' + case_name + '/' + congest_case + '/' + alg_name + '/' + str(agent_flag) + '/' + current_time
if load_data_need:
    save_path += ('_load_data')
if GCN:
    save_path += ('_GCN')
save_path += '/'
if train_or_test == 'test':
    save_path = test_save_path
    save_path = save_path + 'test/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model_save_path = test_save_path + 'model' if train_or_test == 'test' else save_path + 'model'


def get_args():
    """ Hyperparameters
    """
    # curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")

    parser.add_argument('--batch_size',default=128,type=int)

    parser.add_argument('--head_num', default=1, type=int)
    parser.add_argument('--multi_head_fuse', default=False, type=bool)
    parser.add_argument('--GCN_out_dim',default=4,type=int) #64
    parser.add_argument('--GCN_layer',default=1,type=int) #64
    parser.add_argument('--hidden_dim', default=128, type=int) #大于等于64 or None，None不行
    parser.add_argument('--hidden_dim_a', default=None, type=int)  # 默认是None
    parser.add_argument('--start_dim',default=1,type=int)
    parser.add_argument('--num_nodes',default=57,type=int)
    parser.add_argument('--node_embedding',default=20,type=int) #>=5
    parser.add_argument('--reward_scale',default=60,type=float) #30 60

    parser.add_argument('--epsilon', default=1e-10,type=float) #KL or JS
    parser.add_argument('--supports', default=True,type=bool)
    parser.add_argument('--adaptive_GCN', default=True, type=bool)
    parser.add_argument('--GCN_model', default='GWNET', type=str) #GWNET or DGCN DGCN必须dynamic_GCN是True
    parser.add_argument('--time_step', default=3, type=int) #
    parser.add_argument('--TCN_out_dim', default=4, type=int) #如果time_step=1，TCN_out_dim须设置为GCN_out_dim

    parser.add_argument('--save_checkpoint', default=True,type=bool)

    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    # args.device = torch.device("cpu")  # check GPU

    argsDict = args.__dict__
    with open(save_path + '/' + "results_and_paras.txt", "w") as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    return args

cfg = get_args()

n_states = 1 #上时刻的自己的清算价和(action)   注意这个是observation
n_states = n_states+1 if pre_action else n_states
node_num = 4 if case_name == 'case4_disp' else 30 if case_name == 'case30_disp_self' else 57
if full_knowledge is True:
    # n_states = (n_states+n_actions) * n_agents #agent们知道当前其他agent的动作，完美信息，但是好像不符合市场规则了
    # n_states = n_states * n_agents
    n_states = n_states * node_num
n_states = n_states * cfg.time_step
n_states = n_states+1 if max_load_required else n_states


# if train_or_test == 'test':
#     model_save_path = test_save_path
# if not os.path.exists(model_save_path):
#     os.makedirs(model_save_path)

##-----------------------------------二：开始--------------------------------------
eng = matlab.engine.start_matlab()
mpc = eng.loadcase(case_name)
print('case successfully loaded')
bus_id = [int(list(i)[0]) for i in list(mpc['gen'])] #取出各个参与者所在的节点，0就是matlab中mpc.gen矩阵的第一列：bus
branch_num = len(mpc['branch'])
bus_num = len(mpc['bus'])


load_data_default = [100, 200, 120, 320] if case_name == 'case4_disp' else [80, 120, 100] if case_name == 'case30_disp_self' else [0] #genco的包含，已在matlab内定义为最大发电量
# load_data_default = [0.00001, 200, 120, 320] if case_name == 'case4_disp' else [80, 120, 100] if case_name == 'case30_disp_self' else None #genco的包含，已在matlab内定义为最大发电量
# load_data_default = [100, 200, 120, 320] if case_name == 'case4_disp' else [60, 100, 80] #genco的包含，已在matlab内定义为最大发电量
# load_data_default = [100, 200, 120, 320] if case_name == 'case4_disp' else [35, 60, 50] #genco的包含，已在matlab内定义为最大发电量
# load_data_default = [100, 200, 120, 320] if case_name == 'case4_disp' else [45, 75, 60] #genco的包含，已在matlab内定义为最大发电量
# mean_scale = np.array([60, 80, 50])
mean_scale = np.array([60, 80, 50])
# mean_scale = np.array(load_data_default)
if load_data_need:
    # data_size = episode_num*24 #5520个数据，其中最后的test_step个会被视为测试集
    load_data_train_step = load_data_train_episode*24
    # data_path = r'../data/load_data.csv'
    data_path = r'../data/load_data_v2_smooth.csv'
    load_data = pd.read_csv(data_path)
    load_data = load_data.iloc[:,1:1+(n_agents-n_agents_gen)].values
    # load_data = load_data.iloc[:n_steps,1:1+(n_agents-n_agents_gen)].values
    # load_data = load_data.iloc[:data_size,1:1+(n_agents-n_agents_gen)].values

    load_data_temp = load_data[train_day_begin_idx:train_day_begin_idx+(load_data_train_episode+test_episode)*24,:] #默认训练接着测试集
    if test_day_begin_idx:
        load_data_temp[-(test_episode*24):] = load_data[test_day_begin_idx:test_day_begin_idx+test_episode*24, :]
    load_data = load_data_temp
    del load_data_temp

    # mean_scale = load_data.max()
    # load_data_scale = load_data + (mean_scale - load_data.mean(0)) #这个不行，可能出现负数
    scale = (load_data.max(0) / np.array(load_data_default)) * 0.65 #scale一下，将数据的最大值scale到load_data_default中对应load的最大值  0.5
    # scale = load_data.mean(0) / mean_scale #scale一下，将数据的最大值scale到load_data_default中对应load的最大值  0.8
    load_data_scale = load_data / scale
    # scale = (load_data.max(0) / np.array(load_data_default)) * 0.7  # scale一下，将数据的最大值scale到load_data_default中对应load的最大值  0.8
    # load_data_scale = load_data / scale

    load_test_data = load_data_scale[-test_step:,:]

    load_train_data = load_data_scale[:-test_step,:]
    pd.DataFrame(load_data_scale).to_csv(save_path + 'load_data_scale.csv', index=False)
    pd.DataFrame(load_test_data).to_csv(save_path + 'load_data_scale_test.csv', index=False)
    max_load = load_data_scale.sum(1)
    max_load = max_load / max_load.max() #归一化scale一下，再作为state输入到网络，好点
    max_load_test = max_load[-test_step:]
    max_load_train = max_load[:-test_step]
    if train_times == 1:
        load_data_scale_step = load_data_scale
        max_load_step = max_load
    elif train_times > 1:
        if type(train_times) is int:
            load_data_scale_step = np.concatenate(( np.tile(load_train_data, (train_times,1)),load_test_data ))
            max_load_step = np.concatenate(( np.tile(max_load_train, train_times),max_load_test ))
        else:
            integer = int(train_times)
            decimal = train_times % 1
            if integer > 0:
                load_data_scale_step = np.concatenate((np.tile(load_train_data, (integer, 1)), load_train_data[:int(load_data_train_step*decimal),:], load_test_data))
                max_load_step = np.concatenate((np.tile(max_load_train, integer), max_load_train[:int(load_data_train_step*decimal)], max_load_test))
            elif integer == 0:
                load_data_scale_step = np.concatenate((load_train_data[:int(load_data_train_step*decimal),:], load_test_data))
                max_load_step = np.concatenate((max_load_train[:int(load_data_train_step*decimal)], max_load_test))

# else:
#     max_load = 740 #固定load，无意义，不需要考虑

true_gen_a = inverse_linear_map(1, -1, 1, gen_low, gen_high)
true_load_a = inverse_linear_map(1, -1, 1, load_low, load_high)
# true_quantity_a = None
true_quantity_a = inverse_linear_map(1, -1, 1, quantity_low, quantity_high)

##开始训练
experiment_train_returns = []
experiment_strategic_variables = []
experiment_adjacency_a = []
experiment_adjacency_c = []

if load_data_need:
    experiment_test_returns = []
for ex in tqdm(range(experiment_times), desc='experiments'):
    random.seed(sed[ex])
    torch.manual_seed(sed[ex])
    np.random.seed(sed[ex])

    my_list = [25, 53, 49, 40, 33, 27, 21, 31, 11, 30, 45, 26, 28, 19] #bus
    if branch_disc or bus_disc or gen_disc:
        id_dis = np.random.randint(branch_num if branch_disc else bus_num if bus_disc else n_agents, size=dis_num)  # 每次实验模拟dis_num条
        temp = sum(np.isin(id_dis, my_list))
        while temp > 0:
            id_dis = np.random.randint(branch_num if branch_disc else bus_num, size=dis_num)
            temp = sum(np.isin(id_dis, my_list))

        id_dis = np.array([1,2]) #用于自定义，只能对genco调整，retailers用那个load_data_default调整就行

        mpc_ex = disconnect_branch(mpc, id_dis) if branch_disc else disconnect_bus(mpc, id_dis) if bus_disc else disconnect_gen(mpc, id_dis, n_agents_gen)
    else:
        mpc_ex = mpc


    # a = np.zeros(n_agents)
    a = np.zeros((n_agents, n_actions))
    s_ = reset_env(gen_low, gen_high, load_low, load_high, n_agents, n_agents_gen) / cfg.reward_scale  # 是不是有问题，这个状态，状态是指nodal price，不是动作空间！！
    if full_knowledge:
        s_ = np.random.uniform(20, 60, node_num * cfg.time_step) / cfg.reward_scale
    alpha = np.zeros(n_agents)
    beta = np.zeros(n_agents)
    strategic_variables = np.zeros((n_steps, n_agents))
    if quantity_action:
        strategic_variables_beta = np.zeros((n_steps, n_agents))
    rewards = np.zeros((n_steps, n_agents))
    quantity = np.zeros((n_steps, n_agents))
    price = np.zeros((n_steps, n_agents))
    rewards_mean = []
    rewards_agent_mean = []
    rewards_nonagent_mean = []
    rewards_test_agent_mean = []
    quantity_mean = []
    quantity_agent_mean = []
    quantity_nonagent_mean = []
    price_mean = []
    price_agent_mean = []
    price_nonagent_mean = []
    rewards_agent_mean1 = []
    social_welfare = []
    train_returns = []  # 存储着每个回合的平均reward
    test_returns = []  # 存储着每个回合的平均reward
    test_returns_step = []  # 记录数据时间点而已
    pre_a = np.ones(n_agents)
    # a = np.ones(n_agents)
    var_ex = var
    total_load_percentage_all = []

    gencos = []
    for g in range(n_agents):
        # gencos.append(MADDPG(agent_flag[g], alg_name, n_states, n_actions, n_agents, n_agents_gen, n_agents_true, ANet3, CNet3, true_gen_a, true_load_a, MADDPG_critic_input, memory_capacity, gamma=0.90))
        if not GCN: #A2 C2
            gencos.append(MADDPG_true(g, agent_flag[g], alg_name, n_states, n_actions, n_agents, n_agents_gen, n_agents_true,
                                      ANet2, CNet2, true_gen_a, true_load_a, true_quantity_a, MADDPG_critic_input, memory_capacity, gamma=((gamma_default[ex] if type(gamma_default) == list else gamma_default) if auto_gamma is False else auto_gamma_low), lr_a=((lr_default[ex] if type(lr_default) == list else lr_default) if auto_lr is False else auto_lr_high), lr_c=((lr_default[ex] if type(lr_default) == list else lr_default) if auto_lr is False else auto_lr_high), device=device, save_dir=model_save_path, experiment_id=ex if train_or_test == 'train' else test_model_ex_dir[ex], per=PER,
                                      save_rate=evaluate_rate_episode*episode_length, evaluate_begin_step=evaluate_begin_step, episode_length=episode_length, mode=train_or_test, test_round=0 if train_or_test == 'train' else test_round[ex], save_checkpoint=cfg.save_checkpoint, batch_size=cfg.batch_size, soft_rate=(soft_rate[ex] if type(soft_rate) == list else soft_rate) ))
        else:
            gencos.append(MADDPG_GCN(g, agent_flag[g], alg_name, n_states, n_actions, n_agents, n_agents_gen, n_agents_true, ANet_GCN,
                            CNet_GCN, true_gen_a, true_load_a, true_quantity_a, MADDPG_critic_input, memory_capacity, gamma=(gamma_default if auto_gamma is False else auto_gamma_low), lr_a=(lr_default if auto_lr is False else auto_lr_high), lr_c=(lr_default if auto_lr is False else auto_lr_high), device=device, save_dir=model_save_path, experiment_id=ex if train_or_test == 'train' else test_model_ex_dir[ex], per=PER,
                                      save_rate=evaluate_rate_episode*episode_length, evaluate_begin_step=evaluate_begin_step, episode_length=episode_length, mode=train_or_test, test_round=0 if train_or_test == 'train' else test_round[ex], save_checkpoint=cfg.save_checkpoint, batch_size=cfg.batch_size, cfg=cfg))


    replay_buffer_all = [np.zeros_like(gencos[0].memory)] * n_agents

    # for step in tqdm(range(n_steps), desc='steps', leave=False):  # 这个好像就是训练一个回合，但是回合步数非常多，而不是平常那样多回合
    for step in range(0 if train_or_test == 'train' else (n_steps-test_step), n_steps):  # 这个好像就是训练一个回合，但是回合步数非常多，而不是平常那样多回合
        if train_or_test == 'test':
            var_ex = 0
        s = s_ #(8,) ndarray
        for g in range(n_agents):
            # a[g] = gencos[g].choose_action(s)
            # input_s = [s[bus_id[g]]] if full_knowledge is False else list(s)
            input_s = [s[g]] if full_knowledge is False else list(s)
            input_s.append(max_load_step[step]) if max_load_required else None
            (input_s.append(pre_a[g,0]) if full_knowledge is False else input_s.extend(pre_a)) if pre_action else None #上一时刻的动作
            # input_s.extend(a) if full_knowledge is True else None #当前时刻的动作，假设是完美的观测，full knowledge，这个不确定，且未写完

            aa = gencos[g].choose_action(input_s).cpu() #目测主要是因为这一步，就算网络更新和传播使用GPU，也要出来在这转为CPU，所以转换非常耗时。同时因为ac架构，各种输入有耦合，没办法单独给某个网络使用cpu或gpu
            if g < n_agents_gen:
                a[g,0] = np.clip(aa[0] + np.random.randn(1) * var_ex, -1, 1) if gencos[g].agent_flag == 1 else true_gen_a #(n_agent,n_action) ndarray #为什么这句返回的a都是ndarray？？明明是tensor
                alpha[g] = linear_map(a[g,0], -1, 1, gen_low, gen_high) if gencos[g].agent_flag == 1 else 1  # 映射到需要的动作范围
                # if quantity_action:
                a[g,1] = np.clip(aa[1] + np.random.randn(1) * var_ex, -1, 1) if quantity_flag[g] == 1 else true_quantity_a  #为什么这句返回的a都是ndarray？？明明是tensor
                beta[g] = linear_map(a[g,1], -1, 1, quantity_low, quantity_high) if quantity_flag[g] == 1 else 1  # 映射到需要的动作范围
            else:
                # a[g] = np.clip(gencos[g].choose_action(input_s) + np.random.randn(1) * var_ex, -1, 1) if gencos[g].agent_flag == 1 else true_load_a
                # alpha[g] = linear_map(a[g], -1, 1, load_low, load_high) if gencos[g].agent_flag == 1 else 1  # 1就是真实报价
                a[g,0] = np.clip(aa[0] + np.random.randn(1) * var_ex, -1, 1) if gencos[g].agent_flag == 1 else true_load_a
                alpha[g] = linear_map(a[g,0], -1, 1, load_low, load_high) if gencos[g].agent_flag == 1 else 1  # 1就是真实报价
                # if quantity_action:
                a[g,1] = np.clip(aa[1] + np.random.randn(1) * var_ex, -1, 1) if quantity_flag[g] == 1 else true_quantity_a  #为什么这句返回的a都是ndarray？？明明是tensor
                beta[g] = linear_map(a[g,1], -1, 1, quantity_low, quantity_high) if quantity_flag[g] == 1 else 1  # 映射到需要的动作范围

        # market_input = np.concatenate((alpha,beta)) if quantity_action else alpha
        market_input = np.concatenate((alpha,beta))

        # if step < n_steps - test_step: #training phase
        #     data_step = step if step < load_data_train_step else (step % load_data_train_step) #用data_step而不用step是因为这样方便训练完一轮数据集，可以直接重头再继续喂入数据训练，毕竟数据不是无限的
        # elif step >= n_steps-test_step: #test phase
        #     data_step = step % data_size
        # data_step = step if step < load_data_train_step else (step % load_data_train_step)  # 用data_step而不用step是因为这样方便训练完一轮数据集，可以直接重头再继续喂入数据训练，毕竟数据不是无限的
        # nodal_price, profit, obj_value, quant = market_clearing(market_input, (load_data_scale[data_step, :] if load_data_need else load_data_default), eng, mpc_ex, case_name=case_name, verbose=1 if (step + 1 == n_steps) or ((step + 1 >= memory_capacity) & ((step + 1) % 2000 == 0)) else 0)
        nodal_price, profit, obj_value, quant, f, nodal_price_all, total_load_percentage = market_clearing(market_input, (load_data_scale_step[step, :] if load_data_need else load_data_default), eng, mpc_ex, case_name=case_name, verbose=1 if (step + 1 == n_steps) or ((step >= memory_capacity) & ((step ) % 2000 == 0)) else 0)
        # print(np.round(nodal_price_all,2), round(np.mean(nodal_price_all),2))

        strategic_variables[step] = alpha
        if quantity_action:
            strategic_variables_beta[step] = beta
        # r = np.array(profit) / 1000  # 为啥除以1000
        # r = np.ones_like(profit)*-9999 / 1000 if np.isnan(profit).sum()>0 else np.array(profit) / 1000
        r = np.zeros_like(profit) if np.isnan(profit).sum() > 0 else np.array(profit) / 1000
        rewards[step] = r
        rewards_mean.append(np.mean(r)) #np.sum
        rewards_agent_mean.append(np.mean( np.array(r)[agent_id] )) #一步的平均agent reward
        rewards_nonagent_mean.append(np.mean( np.array(r)[nonagent_id] ))
        rewards_agent_mean1.append(np.mean( np.array(r)[agent_id] )) #一步的平均agent reward
        quantity[step] = quant
        quantity_mean.append(np.mean(quant))
        quantity_agent_mean.append(np.mean( np.array(quant)[agent_id] ))
        quantity_nonagent_mean.append(np.mean( np.array(quant)[nonagent_id] ))
        price[step] = nodal_price
        price_mean.append(np.mean(nodal_price))
        price_agent_mean.append(np.mean( np.array(nodal_price)[agent_id] ))
        price_nonagent_mean.append(np.mean( np.array(nodal_price)[nonagent_id] ))
        social_welfare.append(f)
        total_load_percentage_all.append(total_load_percentage)


        # for g in range(n_agents): #fix bug
        if full_knowledge:
            temp = reset_env(gen_low,gen_high,load_low,load_high,n_agents,n_agents_gen)/ cfg.reward_scale if np.isnan(nodal_price_all).sum()>0 else np.array(nodal_price_all) / cfg.reward_scale
            s_ = np.concatenate((s_, temp))
            s_ = s_[node_num:]
        else:
            temp = reset_env(gen_low, gen_high, load_low, load_high, n_agents, n_agents_gen) / cfg.reward_scale if np.isnan(nodal_price).sum() > 0 else np.array(nodal_price) / cfg.reward_scale
            s_ = np.concatenate((s_, temp))
            s_ = s_[n_agents:]

        for g in range(n_agents): #fix bug
            input_s_ = [s_[g]] if full_knowledge is False else list(s_)
            # input_s_ = [s_[bus_id[g]]] if full_knowledge is False else list(s_)
            # input_s_.append(max_load[min(step+1,n_steps-1)]) if max_load_required else None
            input_s_.append(max_load_step[min(step+1,n_steps-1)]) if max_load_required else None
            (input_s_.append(a[g,0]) if full_knowledge is False else input_s_.extend(a)) if pre_action else None
            # input_s_.extend(a) if full_knowledge is True else None #
            # if step < n_steps - test_step:
            if step < n_steps:
                if not PER:
                    gencos[g].store_transition(input_s, a[g], r[g], input_s_)
                else:
                    gencos[g].store_experience(input_s, a[g], r[g], input_s_, 0)
                # gencos[g].store_transition(input_s, a[g], r[g], input_s_) if full_knowledge is False else gencos[g].store_transition(input_s, a, r[g], input_s_)
                # gencos[g].store_transition(s, a[g], r[g], s_)

        # if step >= memory_capacity: #一旦各个agent的replay buffer填完，就开始实时每轮更新replay_buffer_all
        if memory_capacity <= step < n_steps - test_step:
            for i in range(n_agents):
                replay_buffer_all[i] = gencos[i].memory

        if sum(agent_flag) > 0:  # 有agent才进行训练
            if memory_capacity <= step < n_steps-test_step:  # 1000是replay buffer size，9000之后开始测试（阶段性）
                for g in range(n_agents):
                    if gencos[g].agent_flag == 1:
                        if auto_gamma: #递增gamma
                            gencos[g].gamma = min(auto_gamma_high - auto_gamma_decay * (auto_gamma_high - gencos[g].gamma), auto_gamma_high)
                        if auto_lr: #衰减lr
                            # gencos[g].atrain.param_groups[0]['lr'] = max(gencos[g].atrain.param_groups[0]['lr'] * auto_lr_decay, auto_lr_low)
                            # gencos[g].ctrain.param_groups[0]['lr'] = max(gencos[g].ctrain.param_groups[0]['lr'] * auto_lr_decay, auto_lr_low)
                            tep = linear_map(gencos[g].gamma, auto_gamma_low, auto_gamma_high, auto_lr_high, auto_lr_low)  # 自动根据gamma调整lr，逆序映射，大到小，小到大
                            gencos[g].atrain.param_groups[0]['lr'] = tep
                            gencos[g].ctrain.param_groups[0]['lr'] = tep
                        gencos[g].learn(memory_all=replay_buffer_all, agents_list=gencos, agent_true_id=agent_true_id)
                if var_ex > var_low:
                    var_ex *= var_decay_rate

            elif step >= n_steps-test_step-20: #!!!!!!!!!!!!!!
                var_ex = 0


        if step == n_steps-1 and train_or_test == 'train' and cfg.save_checkpoint: #保存最后模型
            for g in range(n_agents):
                gencos[g].save_model(step+1)

        # finish episode, reset the environment
        if episode_length is not None:
            if (step) % episode_length == 0 and step > 0: #每走24步，reset，相当于24步为一个回合
                train_returns.append(np.sum(rewards_agent_mean1)) #存储着每个回合的总reward
                # if not load_data_need: #要是是有连续的数据集的话，就没必要随机重置状态了，为了保证仿真数据的真实连续性（暂时持疑），测试发现是对的
                #     s_ = reset_env(gen_low,gen_high,load_low,load_high,n_agents,n_agents_gen)/ cfg.reward_scale #弄完一个回合，就要重置环境
                #     if full_knowledge:
                #         s_ = np.random.uniform(20, 60, node_num) / cfg.reward_scale

                # 以下a如果不想重置,可以直接注释掉
                # if pre_action: #注意此处的设计是，pre_Action和pre_gamma只能用一个True，因为这里没完善代码，用的都是第一列位置，第零列是报价
                #     # a[:n_agents_gen,0] = np.random.uniform(gen_low, gen_high, n_agents_gen) #弄完一个回合，就要重置环境
                #     a[:n_agents_gen,1] = np.random.uniform(-1, 1, n_agents_gen)  #弄完一个回合，就要重置环境
                #     # a[n_agents_gen:,0] = np.random.uniform(load_low, load_high, n_agents - n_agents_gen) #弄完一个回合，就要重置环境
                #     a[n_agents_gen:,1] = np.random.uniform(-1, 1, n_agents - n_agents_gen) #弄完一个回合，就要重置环境
                # if pre_gamma:
                #     # a[:,1] = np.random.uniform(0, 1, n_agents) #弄完一个回合，就要重置环境
                #     a[id_auto_gamma,1] = np.random.uniform(-1, 1, 1) #弄完一个回合，就要重置环境
                rewards_agent_mean1 = [] #清空该回合的每一步的reward

        # if (load_data_need is True) and (memory_capacity <= step < n_steps - test_step) and ((step) % (evaluate_rate_episode*episode_length) == 0 and step > 0): #每走20个回合，在测试集上测试一次
        # if (load_data_need is True) and (memory_capacity <= step) and (train_or_test=='train') and ((step) % (evaluate_rate_episode*episode_length) == 0 and step > 0): #每走20个回合，在测试集上测试一次
        if (load_data_need is True) and (evaluate_begin_step <= step) and (train_or_test=='train') and ((step) % (evaluate_rate_episode*episode_length) == 0 and step > 0): #每走20个回合，在测试集上测试一次
        # if (load_data_need is True) and (memory_capacity <= step) and ((step) % (evaluate_rate_episode*episode_length) == 0 and step > 0): #每走20个回合，在测试集上测试一次
            s_test_ = reset_env(gen_low,gen_high,load_low,load_high,n_agents,n_agents_gen)/ cfg.reward_scale
            if full_knowledge:
                s_test_ = np.random.uniform(20, 60, node_num) / cfg.reward_scale
            for step_test in range(test_step):
                s_test = s_test_
                for g in range(n_agents):
                    # a[g] = gencos[g].choose_action(s)
                    input_s = [s_test[g]] if full_knowledge is False else list(s_test)
                    input_s.append(max_load_test[step_test]) if max_load_required else None
                    (input_s.append(pre_a[g, 0]) if full_knowledge is False else input_s.extend(pre_a)) if pre_action else None  # 上一时刻的动作

                    aa = gencos[g].choose_action(input_s).cpu()
                    if g < n_agents_gen:
                        a[g, 0] = aa[0] if gencos[g].agent_flag == 1 else true_gen_a  # (n_agent,n_action) ndarray #为什么这句返回的a都是ndarray？？明明是tensor
                        alpha[g] = linear_map(a[g, 0], -1, 1, gen_low, gen_high) if gencos[g].agent_flag == 1 else 1  # 映射到需要的动作范围
                        # if quantity_action:
                        a[g, 1] = aa[1] if quantity_flag[g] == 1 else true_quantity_a  # 为什么这句返回的a都是ndarray？？明明是tensor
                        beta[g] = linear_map(a[g, 1], -1, 1, quantity_low, quantity_high) if quantity_flag[g] == 1 else 1  # 映射到需要的动作范围
                    else:
                        # a[g] = np.clip(gencos[g].choose_action(input_s) + np.random.randn(1) * var_ex, -1, 1) if gencos[g].agent_flag == 1 else true_load_a
                        # alpha[g] = linear_map(a[g], -1, 1, load_low, load_high) if gencos[g].agent_flag == 1 else 1  # 1就是真实报价
                        a[g, 0] = aa[0] if gencos[g].agent_flag == 1 else true_load_a
                        alpha[g] = linear_map(a[g, 0], -1, 1, load_low, load_high) if gencos[g].agent_flag == 1 else 1  # 1就是真实报价
                        # if quantity_action:
                        a[g, 1] = aa[1] if quantity_flag[g] == 1 else true_quantity_a  # 为什么这句返回的a都是ndarray？？明明是tensor
                        beta[g] = linear_map(a[g, 1], -1, 1, quantity_low, quantity_high) if quantity_flag[g] == 1 else 1  # 映射到需要的动作范围

                market_input = np.concatenate((alpha, beta))
                nodal_price, profit, obj_value, quant, f, nodal_price_all, total_load_percentage = market_clearing(market_input, (load_test_data[step_test, :] if load_data_need else load_data_default), eng, mpc_ex, case_name=case_name,verbose=0)

                r = np.zeros_like(profit) if np.isnan(profit).sum() > 0 else np.array(profit) / 1000
                rewards_test_agent_mean.append(np.mean(np.array(r)[agent_id]))
                s_test_ = reset_env(gen_low, gen_high, load_low, load_high, n_agents, n_agents_gen) / cfg.reward_scale if np.isnan(nodal_price).sum() > 0 else np.array(nodal_price) / cfg.reward_scale
                if full_knowledge:
                    s_test_ = reset_env(gen_low, gen_high, load_low, load_high, n_agents, n_agents_gen) / cfg.reward_scale if np.isnan(nodal_price_all).sum() > 0 else np.array(nodal_price_all) / cfg.reward_scale
            test_returns_step.append(step)
            test_returns.append(np.sum(rewards_test_agent_mean))  # 存
            rewards_test_agent_mean = []
            # print('episode:', (step//episode_length), 'test_mean_agent_return:', test_returns[-1])
            print('step:', (step), 'test_mean_agent_return:', test_returns[-1])
            print('-'*20, '\n')


        if (step >= memory_capacity) & ((step) % 1000 == 0):
            # print('Step:', step + 1, 'a1: %.2f' % alpha[0], 'a2: %.2f' % alpha[1], 'r1: %.3f' % profit[0],
            #       'r2: %.3f' % profit[1], 'Explore: %.2f' % var)
            print('\nStep:', step, 'Alpha:', alpha, 'Beta:', beta, 'quantity:', quant, 'Profit:',np.array(profit), 'Profit_mean:',np.array(rewards_mean[-1]), 'Profit_agent_mean:',np.array(rewards_agent_mean[-1]), 'Explore: %.2f' % var_ex, 'lr: %.4f' % gencos[0].ctrain.param_groups[0]['lr'], 'gamma: %.3f' % gencos[0].gamma)


        if (step >= memory_capacity) & ((step) % 500 == 0):
            if GCN:
                # if cfg.entropy_compare is not None:
                loss_a_KL, loss_a, loss_c_KL, loss_c = 0,0,0,0
                for g in range(n_agents):
                    if gencos[g].agent_flag == 1:
                        loss_a_KL += gencos[g].loss_a_KL
                        loss_a += gencos[g].loss_a
                        loss_c_KL += gencos[g].loss_c_KL
                        loss_c += gencos[g].loss_c
                print('\nagent mean loss of [loss_a_KL, loss_a, loss_c_KL, loss_c]: ', [round(i/n_agents_true, 2) for i in [loss_a_KL, loss_a, loss_c_KL, loss_c]])
                # print('genco 1 loss of [loss_a_KL, loss_a, loss_c_KL, loss_c]:', [gencos[0].loss_a_KL, gencos[0].loss_a, gencos[0].loss_c_KL, gencos[0].loss_c] )
                if cfg.auto_scale:
                    print('entropy_loss_scale a and c: ', [round(gencos[0].entropy_loss_scale_A_kl,2), round(gencos[0].entropy_loss_scale_C_kl,2)])
            if show_tensorb:
                if ex==0:
                    with writer.as_default():
                        # tf.summary.scalar(alg_name +'_' + str(agent_flag) + "_rewards_sum" + current_time, rewards_mean[-1], step=step)
                        # tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_quantity_sum" + current_time, quantity_mean[-1], step=step)
                        # tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_social_welfare" + current_time, -obj_value, step=step)

                        tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_loss_a_KL_" + current_time, loss_a_KL/n_agents_true, step=step)
                        tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_loss_a_" + current_time, loss_a/n_agents_true, step=step)
                        tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_loss_c_KL_" + current_time, loss_c_KL/n_agents_true, step=step)
                        tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_loss_c_" + current_time, loss_c/n_agents_true, step=step)

                        # for i in range(n_agents_gen):
                        #     tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_genco{}_".format(i) + current_time, strategic_variables[step,i], step=step, description="genco")
                        # for i in range(n_agents - n_agents_gen):
                        #     tf.summary.scalar(alg_name + '_' + str(agent_flag) + "_load{}_".format(i) + current_time, strategic_variables[step, i+n_agents_gen], step=step, description="load")
                        writer.flush()
                else:
                    writer.close()

            training_std = strategic_variables[(step - training_std_length):step, np.where(np.array(agent_flag) == 1)[0]].std(0).mean() if training_stop_judge == 'action' else np.std(rewards_agent_mean[-training_std_length:])  # 注意np.where输入需要是ndarray，不要list
            print('training std:', training_std)

        pre_a = a


    if GCN and cfg.adaptive_GCN and train_or_test == 'train':
        adp_a = [gencos[g].Actor_eval.adp.detach().mean(dim=0).cpu().numpy() for g in range(n_agents) if gencos[g].agent_flag == 1]
        adp_c = [gencos[g].Critic_eval.adp.detach().mean(dim=0).cpu().numpy() for g in range(n_agents) if gencos[g].agent_flag == 1]
        experiment_adjacency_a.append(adp_a)
        experiment_adjacency_c.append(adp_c)

    experiment_train_returns.append(train_returns) #这个list的元素是某次实验的完整的episode return
    if load_data_need:
        experiment_test_returns.append(test_returns) #这个list的元素是某次实验的完整的episode return


    # save to csv
    if train_or_test == 'train':
        df1 = pd.DataFrame(strategic_variables)
        df2 = pd.DataFrame(rewards)
        df3 = pd.DataFrame(quantity)
        df4 = pd.DataFrame(price)
        df5 = pd.DataFrame(np.concatenate((np.array(rewards_agent_mean).reshape(-1, 1), np.array(rewards_nonagent_mean).reshape(-1, 1), np.array(rewards_mean).reshape(-1, 1)), axis=1))
        df6 = pd.DataFrame(np.concatenate((np.array(price_agent_mean).reshape(-1, 1), np.array(price_nonagent_mean).reshape(-1, 1), np.array(price_mean).reshape(-1, 1)), axis=1))
        df7 = pd.DataFrame(np.concatenate((np.array(quantity_agent_mean).reshape(-1, 1), np.array(quantity_nonagent_mean).reshape(-1, 1), np.array(quantity_mean).reshape(-1, 1)), axis=1))
        if quantity_action:
            df8 = pd.DataFrame(strategic_variables_beta)
        if load_data_need:
            df9 = pd.DataFrame(np.array(rewards_test_agent_mean).reshape(-1, 1))
            # df10 = pd.DataFrame(np.array(test_returns).reshape(-1, 1))
        df10 = pd.DataFrame(social_welfare)
        df11 = pd.DataFrame(total_load_percentage_all)
    else:
        df1 = pd.DataFrame(strategic_variables[-test_step:,:])
        df2 = pd.DataFrame(rewards[-test_step:,:])
        df3 = pd.DataFrame(quantity[-test_step:,:])
        df4 = pd.DataFrame(price[-test_step:,:])
        df5 = pd.DataFrame(np.concatenate((np.array(rewards_agent_mean).reshape(-1, 1), np.array(rewards_nonagent_mean).reshape(-1, 1), np.array(rewards_mean).reshape(-1, 1)), axis=1)[-test_step:,:])
        df6 = pd.DataFrame(np.concatenate((np.array(price_agent_mean).reshape(-1, 1), np.array(price_nonagent_mean).reshape(-1, 1), np.array(price_mean).reshape(-1, 1)), axis=1)[-test_step:,:])
        df7 = pd.DataFrame(np.concatenate((np.array(quantity_agent_mean).reshape(-1, 1), np.array(quantity_nonagent_mean).reshape(-1, 1), np.array(quantity_mean).reshape(-1, 1)), axis=1)[-test_step:,:])
        if quantity_action:
            df8 = pd.DataFrame(strategic_variables_beta[-test_step:,:])
        if load_data_need:
            df9 = pd.DataFrame(np.array(rewards_test_agent_mean).reshape(-1, 1)[-test_step:,:])
            # df10 = pd.DataFrame(np.array(test_returns).reshape(-1, 1))
        df10 = pd.DataFrame(social_welfare[-test_step:])
        df11 = pd.DataFrame(total_load_percentage_all[-test_step:])

    if ex == 0:
        df1.to_csv(save_path + 'result_action.csv', index=False)
        df2.to_csv(save_path + 'result_reward.csv', index=False)
        df3.to_csv(save_path + 'result_quantity.csv', index=False)
        df4.to_csv(save_path + 'result_price.csv', index=False)
        df5.to_csv(save_path + 'reward_agent_nonagent_mean.csv', index=False)
        df6.to_csv(save_path + 'price_agent_nonagent_mean.csv', index=False)
        df7.to_csv(save_path + 'quantity_agent_nonagent_mean.csv', index=False)
        if quantity_action:
            df8.to_csv(save_path + 'result_action_beta.csv', index=False)
        if load_data_need:
            df9.to_csv(save_path + 'rewards_test_agent_mean.csv', index=False)
            # df10.to_csv(save_path + 'return_test_agent_mean.csv', index=False)
        df10.to_csv(save_path + 'social_welfare.csv', index=False)
        df11.to_csv(save_path + 'load_rate.csv', index=False)
    else:
        df1.to_csv(save_path + 'result_action.csv', index=False, mode='a', header=False) #追加写入
        df2.to_csv(save_path + 'result_reward.csv', index=False, mode='a', header=False)
        df3.to_csv(save_path + 'result_quantity.csv', index=False, mode='a', header=False)
        df4.to_csv(save_path + 'result_price.csv', index=False, mode='a', header=False)
        df5.to_csv(save_path + 'reward_agent_nonagent_mean.csv', index=False, mode='a', header=False)
        df6.to_csv(save_path + 'price_agent_nonagent_mean.csv', index=False, mode='a', header=False)
        df7.to_csv(save_path + 'quantity_agent_nonagent_mean.csv', index=False, mode='a', header=False)
        if quantity_action:
            df8.to_csv(save_path + 'result_action_beta.csv', index=False, mode='a', header=False)
        if load_data_need:
            df9.to_csv(save_path + 'rewards_test_agent_mean.csv', index=False, mode='a', header=False)
            # df10.to_csv(save_path + 'return_test_agent_mean.csv', index=False, mode='a', header=False)
        df10.to_csv(save_path + 'social_welfare.csv', index=False, mode='a', header=False)
        df11.to_csv(save_path + 'load_rate.csv', index=False, mode='a', header=False)


# 保存Profit_agent_mean和训练参数
paras_name = "\t".join(['Profit_agent_mean', 'memory_capacity', 'n_steps', 'var', 'pre_action', 'max_load_required',
                        'MADDPG_critic_input', 'gamma_default', 'auto_gamma', 'auto_lr', 'lr_default', 'PER', 'train_day_begin_idx',
                        'load_data_train_episode', 'test_day_begin_idx', 'test_episode',
                        'full_knowledge', 'GCN', 'load_data_default'])  # "\t".join可以将字符串\t间隔插入list中的各个元素之间，最后返回一个拼接好的str
paras = "\t".join(
    [str(np.array(rewards_agent_mean[-1])), str(memory_capacity), str(n_steps), str(var), str(pre_action),
     str(max_load_required), str(MADDPG_critic_input), str(gamma_default), str(auto_gamma), str(auto_lr), str(lr_default),str(PER),
     str(train_day_begin_idx), str(load_data_train_episode), str(test_day_begin_idx), str(test_episode),
     str(full_knowledge), str(GCN), str(load_data_default)])
with open(save_path + '/' + "results_and_paras.txt", "a") as file:
    file.write("\n")
    file.write(paras_name)
    file.write("\n")
    file.write(paras)
    # file.write("\n\n")

# [(i[j] for j in episode_length)  for i in experiment_train_returns]
# average_episodic_rewards = []
new_list = []
for i in range(len(experiment_train_returns[0])):
    sublist = []
    for j in range(len(experiment_train_returns)):
        sublist.append(experiment_train_returns[j][i])
    new_list.append(sublist)
average_episodic_rewards = [np.mean(i) for i in new_list]
average_episodic_rewards_std = [np.std(i) for i in new_list]

#save to csv
df5 = pd.DataFrame(average_episodic_rewards)
df5.to_csv(save_path + 'average_episodic_rewards.csv',index=False)
df6 = pd.DataFrame(average_episodic_rewards_std)
df6.to_csv(save_path + 'average_episodic_rewards_std.csv',index=False)

if GCN and cfg.adaptive_GCN:
    experiment_average_adjacency_a = np.array(experiment_adjacency_a).mean(0)
    experiment_average_adjacency_c = np.array(experiment_adjacency_c).mean(0)
    np.save(save_path + 'experiment_average_adjacency_a.npy', experiment_average_adjacency_a)
    np.save(save_path + 'experiment_average_adjacency_c.npy', experiment_average_adjacency_c)


plot_experiment(save_path, alg_name, case_name, load_data_need, agent_flag, episode_num if train_or_test=='train' else test_episode, memory_capacity)
