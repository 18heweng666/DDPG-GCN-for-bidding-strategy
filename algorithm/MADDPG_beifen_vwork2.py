import math
import os

import numpy as np
import torch
import torch.nn as nn
import copy
import pandas as pd

from collections import namedtuple, deque
import random
import pytz
from collections import namedtuple
from algorithm.model import compute_matrix_dissimilarity_loss


# 定义经验存储元组
# Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class MADDPG_true:
    def __init__(self, agent_id, agent_flag, model_name, s_dim, a_dim, n_agents, n_agents_gen, n_agents_true, ANet, CNet, true_gen_a, true_load_a, true_gamma_a=None, MADDPG_critic_input=True, memory_capacity=1000, gamma=0.0, lr_a=0.001, lr_c=0.001, per=False, device='cpu', save_rate=480, evaluate_begin_step=2000, episode_length=24, save_dir = './model', experiment_id=0, batch_size=128, mode='train', test_round=3120, save_checkpoint=False, soft_rate=0.01): #lr:0.001
        self.agent_id = agent_id
        self.device = device
        self.agent_flag = agent_flag
        self.model_name = model_name
        self.true_gen_a, self.true_load_a, self.true_gamma_a = true_gen_a, true_load_a, true_gamma_a
        self.a_dim, self.s_dim, self.n_agent, self.n_agents_gen = a_dim, s_dim, n_agents, n_agents_gen
        self.gamma = gamma
        self.soft_rate = soft_rate
        self.batch_size = batch_size
        self.Actor_eval = ANet(s_dim, a_dim).to(device)
        self.Actor_target = ANet(s_dim, a_dim).to(device)
        self.MADDPG_critic_input = MADDPG_critic_input
        if model_name == 'MADDPG' and MADDPG_critic_input: #将所有agent或非agent的状态和动作都考虑给agent的critic输入
            self.Critic_eval = CNet(s_dim*n_agents, a_dim*n_agents).to(device) #注意不是：self.Critic_eval = CNet((s_dim+a_dim)*n_agents, a_dim)，self：个人觉得这是更符合直觉的，但是如果从定义的网络结构来看，其实应该是不影响的，无非就是将全部东西丢进去，得到一个标量
            self.Critic_target = CNet(s_dim*n_agents, a_dim*n_agents).to(device)
        elif model_name == 'MADDPG' and (not MADDPG_critic_input): #仅将所有agent的状态和动作考虑给agent的critic输入
            self.Critic_eval = CNet(s_dim*n_agents_true, a_dim*n_agents_true).to(device) #注意不是：self.Critic_eval = CNet((s_dim+a_dim)*n_agents, a_dim)，self：个人觉得这是更符合直觉的，但是如果从定义的网络结构来看，其实应该是不影响的，无非就是将全部东西丢进去，得到一个标量
            self.Critic_target = CNet(s_dim*n_agents_true, a_dim*n_agents_true).to(device)
        elif model_name == 'DDPG':
            self.Critic_eval = CNet(s_dim, a_dim).to(device)
            self.Critic_target = CNet(s_dim, a_dim).to(device)

        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=lr_a)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=lr_c)
        self.loss_td = nn.MSELoss()

        self.train_step = memory_capacity  #0
        self.save_rate = save_rate
        self.evaluate_begin_step = evaluate_begin_step
        self.episode_length = episode_length
        self.save_dir = save_dir
        self.experiment_id = experiment_id
        self.mode = mode
        self.save_checkpoint = save_checkpoint

        self.memory_capacity, self.per = memory_capacity, per
        if self.per:
            self.memory = ReplayTree(self.memory_capacity)  # 用PER
        else:
            self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32) #这两句用于普通buffer
            self.pointer = 0


        # model_path = self.save_dir
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        self.model_path = os.path.join(self.save_dir, 'ex_%d' % self.experiment_id, 'agent_%d' % self.agent_id)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # torch.save(self.Actor_eval.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        # torch.save(self.Critic_eval.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

        # 加载模型
        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        # mode
        # ex = 'ex_8'
        # test_round = 3120
        # x = self.model_path.split('\\')
        # x[-2] = ex
        # model_load_path = os.path.join(x[0],x[1],x[2])
        if self.mode == 'test' and agent_flag == 1:
            self.Actor_eval.load_state_dict(torch.load(self.model_path + '/{}_actor_params.pkl'.format(test_round)))
            self.Critic_eval.load_state_dict(torch.load(self.model_path + '/{}_critic_params.pkl'.format(test_round)))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/{}_actor_params.pkl'.format(test_round)))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/{}_critic_params.pkl'.format(test_round)))


    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(self.device)
        return self.Actor_eval(s)[0].detach()

    def learn(self, memory_all, agents_list, agent_true_id): #0.01
        tau = self.soft_rate

    #以下这段代码实现了Actor-Critic算法中的软更新（soft update），目的是让目标网络（target network）慢慢地接近于当前的估值网络（evaluation network），同时减少训练中的震荡。具体实现过程如下：
    # 对于Actor_target和Critic_target网络中的每一个参数，将其乘以 (1 - tau) 的权重，表示目标网络中的该参数只占原参数的一小部分，大部分还是来自目标网络之前的值。
    # 对于Actor_eval和Critic_eval网络中的每一个参数，将其乘以 tau 的权重，表示目标网络中的该参数主要来自于估值网络的值，同时也加入了少量的目标网络之前的值。
    # 将两个步骤得到的结果相加，得到新的目标网络参数。
    # 其中，tau 是一个较小的常数，用于控制目标网络每次更新时的学习速率，一般设为 0.001 或更小的值。eval() 函数用于将字符串转化为代码进行执行，从而实现对网络中的参数进行访问和修改.
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Actor_target.' + x + '.data.add_(tau*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Critic_target.' + x + '.data.add_(tau*self.Critic_eval.' + x + '.data)')

        if self.per:
            experiences, indices, weights, data_indices = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
            # batch = Experience(*zip(*experiences))
            # states = torch.stack(batch.state)
            # actions = torch.stack(batch.action)
            # rewards = torch.stack(batch.reward)
            # next_states = torch.stack(batch.next_state)
            # dones = torch.stack(batch.done)

            bt = torch.FloatTensor(np.array(experiences)).to(self.device)
            # bt = torch.FloatTensor(self.memory[indices, :]).to(self.device)  # (128,5) tensor
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]
        else:
            indices = np.random.choice(self.memory_capacity, self.batch_size) #(128,)
            # bt = self.memory[indices, :] #(128,5) ndarray
            bt = torch.FloatTensor(self.memory[indices, :]).to(self.device) #(128,5) tensor
            bs = bt[:, :self.s_dim] #(128,1) tensor
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]
            # bs = torch.FloatTensor(bt[:, :self.s_dim]).to(self.device)
            # ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(self.device)
            # br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(self.device)
            # bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(self.device)

        # bt_all = [i[indices,:] for i in memory_all]
        # bs_all = np.array([i[:,:self.s_dim] for i in bt_all]).reshape(self.batch_size, -1)
        # ba_all = np.array([i[:,self.s_dim: self.s_dim + self.a_dim] for i in bt_all]).reshape(self.batch_size, -1)
        # bs_all_ = np.array([i[:, -self.s_dim:] for i in bt_all]).reshape(self.batch_size, -1)
        if self.model_name == 'MADDPG':
            if self.MADDPG_critic_input:
                # memory_all list, member:(1000,5) ndarray
                if not self.per:
                    # bt_all = [i[indices, :] for i in memory_all] # 这个是把非agent的那些真实报价也会考虑到critic的输入，但事实不应该这样，咱们应该只将真agent（flag为1）的那些动作和状态输入给后面的critic
                    bt_all = [torch.FloatTensor(i[indices, :]) for i in memory_all] # 这个是把非agent的那些真实报价也会考虑到critic的输入，但事实不应该这样，咱们应该只将真agent（flag为1）的那些动作和状态输入给后面的critic
                else:
                    bt_all = [torch.FloatTensor(np.vstack(memory.tree.data)[data_indices, :]) for memory in memory_all]
            else:
                if not self.per:
                    # bt_all = [g[indices,:] for id,g in enumerate(memory_all) if agents_list[id].agent_flag == 1] #这个是专门针对agent，把非agent当做环境了相当于
                    bt_all = [torch.FloatTensor(g[indices,:]) for id,g in enumerate(memory_all) if agents_list[id].agent_flag == 1] #这个是专门针对agent，把非agent当做环境了相当于
                #注：agent_true_id就是上行if agents_list[id].agent_flag == 1里的id
                else:
                    bt_all = [torch.FloatTensor(np.vstack(g.tree.data)[data_indices,:]) for id,g in enumerate(memory_all) if agents_list[id].agent_flag == 1]

            bt_all = [t.to(self.device) for t in bt_all] #这样可以发挥gpu的并行计算优势，即并行从cpu转到gpu
            # bt_all = torch.stack(bt_all).to(self.device) #将张量列表整个通过stack变为tensor再to。最快，只进行一次to，但是因为后面要用不少list，所以不好改

            # bs_all = [torch.FloatTensor(i[:,:self.s_dim]).to(self.device) for i in bt_all]
            # ba_all = [torch.FloatTensor(i[:,self.s_dim: self.s_dim + self.a_dim]).to(self.device) for i in bt_all]
            # bs_all_ = [torch.FloatTensor(i[:, -self.s_dim:]).to(self.device) for i in bt_all]
            bs_all = [i[:,:self.s_dim] for i in bt_all]
            ba_all = [i[:,self.s_dim: self.s_dim + self.a_dim] for i in bt_all]
            bs_all_ = [i[:, -self.s_dim:] for i in bt_all]

            with torch.no_grad():
                # ba_all_ = [agent.Actor_target(torch.FloatTensor(bs_all_[id])).numpy() for id,agent in enumerate(agents_list) if agent.agent_flag == 1 ] #用.numpy()还是.detach()
                if self.MADDPG_critic_input:
                    if self.true_gamma_a is not None:
                        ba_all_ = [ (agent.Actor_target(bs_all_[id]) if agent.agent_flag == 1 else (torch.ones((self.batch_size,self.a_dim))*torch.tensor([self.true_gen_a,self.true_gamma_a]) if id < self.n_agents_gen else torch.ones((self.batch_size,self.a_dim))*torch.tensor([self.true_load_a,self.true_gamma_a]))) for id,agent in enumerate(agents_list)] # 这个是把非agent的那些真实报价也会考虑到critic的输入，但事实不应该这样，咱们应该只将真agent（flag为1）的那些动作和状态输入给后面的critic
                    else:
                        ba_all_ = [ (agent.Actor_target(bs_all_[id]) if agent.agent_flag == 1 else (torch.ones((self.batch_size,self.a_dim))*self.true_gen_a if id < self.n_agents_gen else torch.ones((self.batch_size,self.a_dim))*self.true_load_a) ) for id,agent in enumerate(agents_list)] # 这个是把非agent的那些真实报价也会考虑到critic的输入，但事实不应该这样，咱们应该只将真agent（flag为1）的那些动作和状态输入给后面的critic
                else:
                    ba_all_ = [agents_list[g].Actor_target(bs_all_[i]) for i,g in enumerate(agent_true_id)]

            if self.MADDPG_critic_input:
                # agent_id是当前要训练的agent id，从agent_flag中来。比如长度为8，8个里只有部分为agent
                actor_input_ba_all = copy.deepcopy(ba_all)  # 要copy一下，否则经过下一行的传播，actor_input_ba_all带入的梯度会使得ba_all也带入了梯度
                actor_input_ba_all[self.agent_id] = self.Actor_eval(bs)  # 类似于maddpg代码中的u[self.self.agent_id] = self.actor_network(o[self.self.agent_id])
            else:
                # self.agent_id是当前要训练的agent id，从agent_true_id中来，比如长度为4,4个都是真agent
                idx = int(np.where(np.array(agent_true_id) == self.agent_id)[0])
                actor_input_ba_all = copy.deepcopy(ba_all)  # 要copy一下，否则经过下一行的传播，actor_input_ba_all带入的梯度会使得ba_all也带入了梯度
                actor_input_ba_all[idx] = self.Actor_eval(bs)  # 类似于maddpg代码中的u[self.self.agent_id] = self.actor_network(o[self.self.agent_id])


            bs_all = torch.cat(bs_all,dim=1) #会将list自动变成tensor
            ba_all = torch.cat(ba_all,dim=1)
            bs_all_ = torch.cat(bs_all_,dim=1)
            ba_all_ = [t.to(self.device) for t in ba_all_] #因为有些agent不是真agent，是创建在cpu上的tensor，所以在这先统一转到gpu
            ba_all_ = torch.cat(ba_all_,dim=1)
            actor_input_ba_all = torch.cat(actor_input_ba_all,dim=1)

            # 更新critic
            q_ = self.Critic_target(bs_all_, ba_all_).detach()
            q_v = self.Critic_eval(bs_all, ba_all)
            # with torch.no_grad():
            q_target = br + self.gamma * q_
            # q_target = br + self.gamma * q_

            # if self.per and ((self.train_step % 1000 == 0) or self.train_step == 250 or self.train_step == 500) and (self.mode == 'train'):
            #     experience_curr = self.memory.get_experience_data()
            #     experience_curr.to_csv(self.model_path + '/' + 'experience.csv', mode='a', header=False, index=False)

            if self.per:
                td_errors = q_target - q_v
                self.memory.batch_update(indices, td_errors.cpu().detach().numpy())
                loss_c = (weights * td_errors.pow(2)).mean() #critic_loss
            else:
                loss_c = self.loss_td(q_target, q_v)

            # self.ctrain.zero_grad()
            # loss_c.backward()
            # self.ctrain.step()

            # 更新actor
            # q = self.Critic_eval(bs_all, ba_all)
            q = self.Critic_eval(bs_all, actor_input_ba_all)
            loss_a = -torch.mean(q)
            # self.atrain.zero_grad()
            # loss_a.backward()
            # self.atrain.step()


        elif self.model_name == 'DDPG':
            #更新actor
            a = self.Actor_eval(bs)
            q = self.Critic_eval(bs, a)
            # 更新actor
            loss_a = -torch.mean(q)
            # self.atrain.zero_grad()
            # loss_a.backward()
            # self.atrain.step()

            #更新critic
            a_ = self.Actor_target(bs_)
            q_ = self.Critic_target(bs_, a_)
            q_v = self.Critic_eval(bs, ba)

            q_target = br + self.gamma * q_
            loss_c = self.loss_td(q_target, q_v)
            # self.ctrain.zero_grad()
            # loss_c.backward()
            # self.ctrain.step()


        #ddpg和maddpg都一样，注意这里不能先对ctrain再对atrain更新，会报错，是因为pytorch语法问题，但是如果按照前面的写法分开写，是可以的
        self.atrain.zero_grad()
        loss_a.backward() #actor
        self.atrain.step()
        self.ctrain.zero_grad()
        loss_c.backward() #critic
        self.ctrain.step()

        # if self.train_step > 0 and self.train_step % self.save_rate == 0:
        # if (self.train_step >= self.memory_capacity) and ((self.train_step) % self.save_rate == 0 and self.train_step > 0):
        # if self.save_checkpoint and (self.train_step >= self.memory_capacity) and (self.train_step % self.save_rate == 0) and (self.mode == 'train'):
        if self.save_checkpoint and (self.train_step >= self.evaluate_begin_step) and (self.train_step % self.save_rate == 0) and (self.mode == 'train'):
            self.save_model(self.train_step)


        # if self.per and (self.train_step % 1000 == 0) and (self.mode == 'train'):
        #     experience_curr = self.memory.get_experience_data()
        #     experience_curr.to_csv(self.model_path + '/' + 'experience.csv', mode='a', header=False, index=False)

        # if self.per and ((self.train_step % 1000 == 0) or (self.train_step == 250) or (self.train_step == 500) or (self.train_step == 200*24-5 )) and (self.mode == 'train'):
        #     experience_curr = self.memory.get_experience_data()
        #     experience_curr.to_csv(self.model_path + '/' + 'experience.csv', mode='a', header=False, index=False)

        self.train_step += 1

    def save_model(self, train_step):
        # num = str(train_step // self.save_rate)
        # num = str(train_step // self.episode_length)
        num = str(train_step)
        # model_path = self.save_dir
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)

        torch.save(self.Actor_eval.state_dict(), self.model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.Critic_eval.state_dict(),  self.model_path + '/' + num + '_critic_params.pkl')
        # print('Agent {}'.format(self.agent_id), 'checkpoint episode', num, 'saved')
        print('Agent {}'.format(self.agent_id), 'checkpoint step', num, 'saved')

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def store_experience(self, s, a, r, s_, done): #PER
        # state = torch.tensor(state, dtype=torch.float32)
        # action = torch.tensor(action, dtype=torch.float32)
        # reward = torch.tensor(reward, dtype=torch.float32)
        # next_state = torch.tensor(next_state, dtype=torch.float32)
        # done = torch.tensor(done, dtype=torch.float32)

        # experience = Experience(state, action, reward, next_state, done)
        experience = np.hstack((s, a, [r], s_))
        self.memory.push(experience, 1)




class MADDPG_GCN:
    def __init__(self, agent_id, agent_flag, model_name, s_dim, a_dim, n_agents, n_agents_gen, n_agents_true, ANet, CNet, true_gen_a, true_load_a, true_gamma_a=None, MADDPG_critic_input=True, memory_capacity=1000, gamma=0.0, lr_a=0.001, lr_c=0.001, per=False, device='cpu', save_rate=480, evaluate_begin_step=2000, episode_length=24, save_dir = './model', experiment_id=0, batch_size=128, mode='train', test_round=3120, save_checkpoint=False, cfg=None): #lr:0.001
        self.adaptive_GCN = cfg.adaptive_GCN
        self.agent_id = agent_id
        self.device = device
        self.agent_flag = agent_flag
        self.model_name = model_name
        self.true_gen_a, self.true_load_a, self.true_gamma_a = true_gen_a, true_load_a, true_gamma_a
        self.a_dim, self.s_dim, self.n_agent, self.n_agents_gen = a_dim, s_dim, n_agents, n_agents_gen
        self.gamma = gamma
        self.batch_size = batch_size

        #如果是DDPG，那就是GCN+DDPG，第二篇工作；否则就是GCN+MADDPG，第三篇工作
        if model_name == 'DDPG':
            self.Actor_eval = ANet(s_dim, a_dim, cfg, device=device).to(device)
            self.Actor_target = ANet(s_dim, a_dim, cfg, device=device).to(device)
        elif model_name == 'MADDPG':
            self.Actor_eval = ANet(s_dim, a_dim, cfg=cfg).to(device)
            self.Actor_target = ANet(s_dim, a_dim, cfg=cfg).to(device)

        self.MADDPG_critic_input = MADDPG_critic_input


        # self.epsilon = cfg.epsilon
        self.loss_a, self.loss_c, self.loss_a_KL, self.loss_c_KL = 0,0,0,0


        if model_name == 'MADDPG' and MADDPG_critic_input: #将所有agent或非agent的状态和动作都考虑给agent的critic输入
            self.Critic_eval = CNet(s_dim*n_agents, a_dim*n_agents, cfg, device=device).to(device) #注意不是：self.Critic_eval = CNet((s_dim+a_dim)*n_agents, a_dim)，self：个人觉得这是更符合直觉的，但是如果从定义的网络结构来看，其实应该是不影响的，无非就是将全部东西丢进去，得到一个标量
            self.Critic_target = CNet(s_dim*n_agents, a_dim*n_agents, cfg, device=device).to(device)
        elif model_name == 'MADDPG' and (not MADDPG_critic_input): #仅将所有agent的状态和动作考虑给agent的critic输入
            self.Critic_eval = CNet(s_dim*n_agents_true, a_dim*n_agents_true).to(device) #注意不是：self.Critic_eval = CNet((s_dim+a_dim)*n_agents, a_dim)，self：个人觉得这是更符合直觉的，但是如果从定义的网络结构来看，其实应该是不影响的，无非就是将全部东西丢进去，得到一个标量
            self.Critic_target = CNet(s_dim*n_agents_true, a_dim*n_agents_true).to(device)
        elif model_name == 'DDPG':
            self.Critic_eval = CNet(s_dim, a_dim, cfg, device=device).to(device)
            self.Critic_target = CNet(s_dim, a_dim, cfg, device=device).to(device)

        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=lr_a)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=lr_c)
        # self.atrain = torch.optim.RMSprop(self.Actor_eval.parameters(), lr=lr_a)
        # self.ctrain = torch.optim.RMSprop(self.Critic_eval.parameters(), lr=lr_c)
        self.loss_td = nn.MSELoss()

        self.train_step = memory_capacity  #0
        self.save_rate = save_rate
        self.evaluate_begin_step = evaluate_begin_step
        self.episode_length = episode_length
        self.save_dir = save_dir
        self.experiment_id = experiment_id
        self.mode = mode
        self.save_checkpoint = save_checkpoint

        self.memory_capacity, self.per = memory_capacity, per


        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32) #这两句用于普通buffer
        self.pointer = 0


        # model_path = self.save_dir
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        self.model_path = os.path.join(self.save_dir, 'ex_%d' % self.experiment_id, 'agent_%d' % self.agent_id)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # torch.save(self.Actor_eval.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        # torch.save(self.Critic_eval.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

        # 加载模型
        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        # mode
        # ex = 'ex_8'
        # test_round = 3120
        # x = self.model_path.split('\\')
        # x[-2] = ex
        # model_load_path = os.path.join(x[0],x[1],x[2])
        if self.mode == 'test' and agent_flag == 1:
            self.Actor_eval.load_state_dict(torch.load(self.model_path + '/{}_actor_params.pkl'.format(test_round)))
            self.Critic_eval.load_state_dict(torch.load(self.model_path + '/{}_critic_params.pkl'.format(test_round)))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/{}_actor_params.pkl'.format(test_round)))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/{}_critic_params.pkl'.format(test_round)))


    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0).to(self.device)
        return self.Actor_eval(s)[0].detach()

    def learn(self, memory_all, agents_list, agent_true_id, tau=0.01): #0.01
    #以下这段代码实现了Actor-Critic算法中的软更新（soft update），目的是让目标网络（target network）慢慢地接近于当前的估值网络（evaluation network），同时减少训练中的震荡。具体实现过程如下：
    # 对于Actor_target和Critic_target网络中的每一个参数，将其乘以 (1 - tau) 的权重，表示目标网络中的该参数只占原参数的一小部分，大部分还是来自目标网络之前的值。
    # 对于Actor_eval和Critic_eval网络中的每一个参数，将其乘以 tau 的权重，表示目标网络中的该参数主要来自于估值网络的值，同时也加入了少量的目标网络之前的值。
    # 将两个步骤得到的结果相加，得到新的目标网络参数。
    # 其中，tau 是一个较小的常数，用于控制目标网络每次更新时的学习速率，一般设为 0.001 或更小的值。eval() 函数用于将字符串转化为代码进行执行，从而实现对网络中的参数进行访问和修改.
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Actor_target.' + x + '.data.add_(tau*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Critic_target.' + x + '.data.add_(tau*self.Critic_eval.' + x + '.data)')

        indices = np.random.choice(self.memory_capacity, self.batch_size) #(128,)
        # bt = self.memory[indices, :] #(128,5) ndarray
        bt = torch.FloatTensor(self.memory[indices, :]).to(self.device) #(128,5) tensor
        bs = bt[:, :self.s_dim] #(128,1) tensor
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        #更新actor
        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)
        # 更新actor
        loss_a = -torch.mean(q)
        self.loss_a = loss_a.item()
        # self.atrain.zero_grad()
        # loss_a.backward()
        # self.atrain.step()

        loss_a_total = loss_a

        #更新critic
        a_ = self.Actor_target(bs_)
        q_ = self.Critic_target(bs_, a_)
        q_v = self.Critic_eval(bs, ba)

        q_target = br + self.gamma * q_
        loss_c = self.loss_td(q_target, q_v)
        self.loss_c = loss_c.item()
        # self.ctrain.zero_grad()
        # loss_c.backward()
        # self.ctrain.step()

        loss_c_total = loss_c



        self.atrain.zero_grad()
        if self.model_name == 'DDPG':
            loss_a_total.backward()  # actor
        else:
            loss_a.backward() #actor
        self.atrain.step()

        self.ctrain.zero_grad()
        if self.model_name == 'DDPG':
            loss_c_total.backward()  # actor
        else:
            loss_c.backward()  # actor
        self.ctrain.step()

        if self.save_checkpoint and (self.train_step >= self.evaluate_begin_step) and (self.train_step % self.save_rate == 0) and (self.mode == 'train'):
            self.save_model(self.train_step)

        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step)
        torch.save(self.Actor_eval.state_dict(), self.model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.Critic_eval.state_dict(),  self.model_path + '/' + num + '_critic_params.pkl')
        # print('Agent {}'.format(self.agent_id), 'checkpoint episode', num, 'saved')
        print('Agent {}'.format(self.agent_id), 'checkpoint step', num, 'saved')

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def store_experience(self, s, a, r, s_, done): #PER
        experience = np.hstack((s, a, [r], s_))
        self.memory.push(experience, 1)