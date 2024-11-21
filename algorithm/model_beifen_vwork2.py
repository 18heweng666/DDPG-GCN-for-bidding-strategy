import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.att import MHSA
from torch.autograd import Variable

# from scipy.stats import wasserstein_distance


class ANet1(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet1, self).__init__()
        self.FC1 = nn.Linear(s_dim, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, a_dim)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))
        return result


class CNet1(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet1, self).__init__()
        self.FC1 = nn.Linear(s_dim, 128)
        self.FC2 = nn.Linear(128 + a_dim, 128)
        self.FC3 = nn.Linear(128, 64)
        self.FC4 = nn.Linear(64, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs)) #为啥要先对obs用FC1变换一下，直接FC2不香吗
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class ANet2(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=64, cfg=None): #64
        super(ANet2,self).__init__()
        # if cfg is not None:
        #     hidden_dim = cfg.
        self.fc1 = nn.Linear(s_dim, hidden_dim)  #64
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class CNet2(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=64, cfg=None, device='cpu'): #64
        super(CNet2,self).__init__()
        self.fcs = nn.Linear(s_dim, hidden_dim) #64
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, hidden_dim)
        self.fca.weight.data.normal_(0, 0.1)
        self.fcsa = nn.Linear(hidden_dim, hidden_dim)
        self.fcsa.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        xy = F.relu(x + y)
        xy = F.relu(self.fcsa(xy))
        actions_value = self.out(xy)
        return actions_value


class ANet25(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=128): #128
        super(ANet25,self).__init__()
        self.fc1 = nn.Linear(s_dim, hidden_dim)  #64
        self.fc1.weight.data.normal_(0, 0.1)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)  #64
        # self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.relu(self.fc2(x))
        x = self.out(x)
        x = torch.tanh(x)
        return x


class CNet25(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=128): #128
        super(CNet25,self).__init__()
        self.fcs = nn.Linear(s_dim, hidden_dim) #64
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, hidden_dim)
        self.fca.weight.data.normal_(0, 0.1)
        self.fcsa = nn.Linear(hidden_dim, hidden_dim)
        self.fcsa.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        xy = F.relu(x + y)
        # xy = F.relu(self.fcsa(xy))
        actions_value = self.out(xy)
        return actions_value


class ANet3(nn.Module):
    def __init__(self, o_dim, a_dim):
        super(ANet3, self).__init__()
        self.fc1 = nn.Linear(o_dim, 64) #注意输入args.obs_shape[agent_id]
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, a_dim)
        # self.bn = nn.BatchNorm1d(o_dim)

    def forward(self, x):
        # x = self.bn(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = torch.tanh(self.action_out(x))
        return actions


class CNet3(nn.Module):
    def __init__(self, o_dim, a_dim):
        super(CNet3, self).__init__()
        self.fc1 = nn.Linear(o_dim+a_dim, 64) #注意输入 sum(args.obs_shape) + sum(args.action_shape)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # self.bn = nn.BatchNorm1d(o_dim+a_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        # x = self.bn(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class ANet4(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet4,self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class CNet4(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet4,self).__init__()
        self.fcs = nn.Linear(s_dim, 64)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 64)
        self.fca.weight.data.normal_(0, 0.1)
        self.fcsa = nn.Linear(64, 128)
        self.fcsa.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        xy = F.relu(x + y)
        xy = F.relu(self.fcsa(xy))
        actions_value = self.out(xy)
        return actions_value


class nconv(nn.Module):
    def __init__(self, dynamic_GCN=False):
        super(nconv,self).__init__()
        self.dynamic_GCN = dynamic_GCN

    def forward(self,x, A):
        # 输入16 32 792 12        1 30 32    30 30   1 30 32
        # A=A.transpose(-1,-2) #转置一下A矩阵(792 792)，好像没什么意义吧？？答：有！！！正常公式是AXW，但从下面矩阵乘法的顺序以及X的维度排序可看出，作者变成了X^T A W,但这样肯定不对的所以要变为X^T A^T W，才与原本公式前两个的整体呈一个转置关系。
        # x = torch.einsum('ncvl,vw->ncwl',(x,A))
        if self.dynamic_GCN:
            # x = torch.einsum('bnm,bmc->bnc',(A,x)) #矩阵乘法AX
            x = torch.einsum('bnm,bmct->bnct',(A,x)) #矩阵乘法AX
        else:
            # x = torch.einsum('nm,bmc->bnc', (A, x))  # 矩阵乘法AX
            x = torch.einsum('nm,bmct->bnct', (A, x))  # 矩阵乘法AX
        return x.contiguous()


class multi_gcn(nn.Module):
    def __init__(self,c_in,c_out,support_len=2,order=1, dynamic_GCN=False):
        super(multi_gcn,self).__init__()
        self.nconv = nconv(dynamic_GCN)
        c_in = (order*support_len+1)*c_in
        # c_in = c_in
        self.mlp = nn.Linear(c_in,c_out)
        # self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=False) #4维时间扩充
        # self.dropout = dropout
        self.order = order

    def forward(self,x,support): #这个x和support哪里输入来的？？？？？
        #答：在model.py中395行：x = self.gconv[i](x, new_supports)，相当于x = multi_gcn(x, new_supports),这里的x, new_supports就传给了multi_gcn的forward输入的参数
        out = [x] #x:16,32,792,12
        for a in support:  #这一段相当于实现GCN中AXW中的AX，但是这个AX用的是gwnet里的“AX”
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1): #order:2,这里循环了一次
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        #out里有7个16,32,792,12矩阵    3个1 30 32 矩阵

        # h = torch.cat(out,dim=1) #某一次进来这个forward：16,224,792,12，另一次：16,32,792,7
        # h = torch.cat(out,dim=-1) #1 30 96    1 30 6
        # h = torch.cat(out,dim=-2) #1 30 96    1 30 6
        # h = torch.cat(out,dim=-2).permute(0, 2, 1, 3) #1 30 96    1 30 6 考虑时间维度扩充   1 4 30 t
        h = torch.cat(out,dim=-2).permute(0, 3, 1, 2) #1 30 96    1 30 6 考虑时间维度扩充   1 4 30 t
        # h = self.mlp(h) #1 30 32
        # h = F.relu(self.mlp(h)) #1 30 32     1 30 1
        # h = F.relu(self.mlp(h)).permute(0, 2, 1, 3) #1 node out_dim t
        h = F.relu(self.mlp(h)).permute(0, 3, 1, 2) #1 node out_dim t
        # 上一行实现应该是有问题的，原文公式三个W是不一样的，但是这里相当于三个W用一个W去搞了！！而且意义不单止是这样。
        #具体是：原文是直接各个相加再对加进行权重映射调整，而这里是直接对所有的各个进行加权求和，目测两者效果应该差不太多，但后者参数量是三分之一的W。
        # if self.dropout is not None:
        #     h = F.dropout(h, self.dropout, training=self.training)
        return h


class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt): #Kt是时间维度的卷积核size，用于提取时间维度特征
        super(T_cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = nn.Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        # x 32 64 170 60   adj 32 170 170
        nSample, feat_in, nNode, length = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)  # 32 3 170 170
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()  # 32 64 3 170 60 相当于分别用3个邻接矩阵进行3次图卷积
        x = x.view(nSample, -1, nNode, length)  # 32 192 170 60
        out = self.conv1(x)  # 32 128 170 60 由于卷积核的设计，这里把时间维度也卷了
        return out


class cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K=1, device='cpu'): #取消了Kt，不包含时间维度卷积
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = nn.Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K
        self.device = device

    def forward(self, x, adj):
        # x 32 64 170 60   adj 32 170 170
        x = x.permute(0, 2, 1, 3)
        nSample, feat_in, nNode, length = x.shape

        Ls = []
        L1 = adj
        # L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)  # 32 3 170 170
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()  # 32 64 3 170 60 相当于分别用3个邻接矩阵进行3次图卷积
        x = x.view(nSample, -1, nNode, length)  # 32 192 170 60
        out = self.conv1(x)  # 32 128 170 60
        return out


class multi_head_A(nn.Module):
    def __init__(self, head=4, node_embedding=10, num_nodes=None, device='cuda', epsilon=1e-8, aggregate=False):
        super(multi_head_A,self).__init__()
        self.head = head
        self.nodevec1 = nn.Parameter(torch.randn(head, num_nodes, node_embedding).to(device), requires_grad=True).to(device) #We randomly initialize node embeddings by a normal distribution with a size of 10.
        self.nodevec2 = nn.Parameter(torch.randn(head, node_embedding, num_nodes).to(device), requires_grad=True).to(device)
        # self.weight_A = nn.Parameter(torch.randn(head).to(device), requires_grad=True).to(device)
        self.epsilon = epsilon
        self.aggregate = aggregate

    def forward(self, input=None):
        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1) #(792,792)，nodevec1(792,10), nodevec1(10,792),注意每次forward完由于更新了参数（包括nodevec1和nodevec2参数），所以每次运行这里所得的adp值都不一样，一直在进步
        # 下面是第二种激活法，将softmax替换为norm行归一化
        # adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))  # (792,792)，nodevec1(792,10), nodevec1(10,792),注意每次forward完由于更新了参数（包括nodevec1和nodevec2参数），所以每次运行这里所得的adp值都不一样，一直在进步

        #法一：softmax
        # adp = F.softmax(F.relu(torch.einsum('hnc,hcm->hnm', (self.nodevec1, self.nodevec2))), dim=-1) #矩阵乘法AX,得到(head, num_nodes, num_nodes)
        #法二：norm
        adp = F.relu(torch.einsum('hnc,hcm->hnm', (self.nodevec1, self.nodevec2))) #矩阵乘法AX,得到(head, num_nodes, num_nodes)
        d = 1 / (torch.sum(adp, -1) + self.epsilon) #1e-8
        # d = 1 / (torch.sum(adp, -1)) #1e-8
        D = torch.diag_embed(d)
        ### adp = torch.matmul(D, adp)
        adp = torch.einsum('hnc,hcm->hnm', (D, adp)) #(head, num_nodes, num_nodes)

        # if input is not None:
        #     adp_input = F.softmax((input.squeeze(0) * input.squeeze(0).T).unsqueeze(0), dim=1)
        #     adp = torch.concat((adp, adp_input), dim=0)

        if self.aggregate:
            adp_agg = torch.mean(adp, dim=0, keepdim=True) #平均融合
            # weight_A = F.softmax(self.weight_A) #自适应融合
            # adp_agg = torch.sum(torch.einsum('hnm,h->hnm', (adp, weight_A)), dim=0, keepdim=True)
            return adp.contiguous(), adp_agg.contiguous()
        else:
            return adp.contiguous()


def load_adj(path):
    import pandas as pd
    load_data = pd.read_csv(path, header=None)
    # return torch.tensor(load_data.values, requires_grad=True).
    return load_data.values

class GRU(nn.Module):
    def __init__(self, length=12, in_dim=1, hidden_dim=32, out_dim=12, device='cuda'):
        super(GRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)  # b*n,l,c
        self.c_out = hidden_dim
        tem_size = length
        self.tem_size = tem_size

        self.conv1 = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, tem_size),
                            stride=(1, 1), bias=True)

    def forward(self, input):
        x = input  # (16,1,792,12)
        shape = x.shape
        h = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).to(self.device)  # 1,12672,32
        hidden = h

        x = x.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])  # 12672,12,1
        x, hidden = self.gru(x,hidden)  # 12672,12,32，注意输入特征1维，输出特征32维。和1,12672，32，hidden就是最后一个gru cell输出的h，总共有12个cell（因为window size是12）
        x = x.view(shape[0], shape[2], shape[3], self.c_out).permute(0, 3, 1, 2).contiguous()  # 16,32,792,12，输入特征从1变为32
        x = self.conv1(x)  # b,c,n,l #16,12,792,1,即特征维度32变12,序列长度12变为1   b d n

        # x = x[:, -1, :].view(shape[0], shape[2], self.c_out)  #b n d'
        # return x, hidden[0], hidden[0]
        return x


class ANet_GCN(nn.Module):
    def __init__(self, o_dim, a_dim, cfg, device='cpu'): #cuda or cpu
        super(ANet_GCN, self).__init__()
        self.o_dim = o_dim
        self.start_dim = cfg.start_dim
        self.GCN_out_dim = cfg.GCN_out_dim
        self.num_nodes = cfg.num_nodes
        self.device = device
        self.adaptive_GCN = cfg.adaptive_GCN
        self.multi_head_fuse = cfg.multi_head_fuse
        self.GCN_layer = cfg.GCN_layer
        self.hidden_dim_a = cfg.hidden_dim_a

        # self.GCN_layer = 1
        # if self.entropy_compare is not None:
        self.adp = None
        self.adp_fuse = None
        if cfg.supports is False:
            self.supports = []
        else:
            # data_path = r'../data/adjacency_P_self_30.csv' if self.num_nodes == 30 else r'../data/adjacency_P_self_57.csv'
            data_path = r'../data/adjacency_P_self_{}.csv'.format(self.num_nodes)
            self.supports = [torch.tensor(load_adj(data_path), requires_grad=False).float().to(device)] #True

        if self.multi_head_fuse:
            head_num_multigcn = 1
        else:
            head_num_multigcn = cfg.head_num
        supports_len = 1 if (cfg.supports is True and cfg.adaptive_GCN is False) \
            else (1 + head_num_multigcn) if (cfg.supports is True and cfg.adaptive_GCN is True) \
            else head_num_multigcn

        if self.dynamic_GCN is True:
            supports_len += 1

        #if (cfg.supports is False and cfg.adaptive_GCN is True)

        self.time_step = cfg.time_step
        self.TCN_out_dim = cfg.TCN_out_dim

        # self.fc0 = nn.Linear(1, self.start_dim)
        self.fc0 = nn.Conv2d(1, self.start_dim, kernel_size=(1, 1), stride=(1, 1), bias=True) #4维向量扩充
        if self.hidden_dim_a is not None:
            self.fc1 = nn.Linear(self.GCN_out_dim * self.num_nodes, self.hidden_dim_a)
            self.action_out = nn.Linear(self.hidden_dim_a, a_dim)
        else:
            # self.action_out = nn.Linear(self.GCN_out_dim * self.num_nodes, a_dim)
            self.action_out = nn.Linear(self.TCN_out_dim * self.num_nodes, a_dim)
            # self.action_out = nn.Linear(self.TCN_out_dim * self.num_nodes * self.time_step, a_dim)

        # self.fc1 = nn.Linear(o_dim, 64) #注意输入args.obs_shape[agent_id]
        # self.fc2 = nn.Linear(64, 64)
        self.GCN_model = cfg.GCN_model
        if self.GCN_model == 'DGCN':
            # self.gconv = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)(self.start_dim, self.GCN_out_dim, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x
            self.gconv = cheby_conv_ds(self.start_dim, self.GCN_out_dim, K=2, device=self.device) #这个K就是multi_gcn里的order+1
        elif self.GCN_model == 'GWNET':
            self.gconv = multi_gcn(self.start_dim, self.GCN_out_dim, order=1, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x

        # self.gconv1 = multi_gcn(self.GCN_out_dim, self.GCN_out_dim, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x
        self.gconv1 = None
        # self.fc3 = nn.Linear(self.GCN_out_dim*self.num_nodes, cfg.hidden_dim)

        # self.action_out = nn.Linear(self.GCN_out_dim*self.num_nodes, a_dim)

        self.multi_head_A = multi_head_A(head=cfg.head_num, node_embedding=cfg.node_embedding, num_nodes=self.num_nodes, device=device, epsilon=cfg.epsilon, aggregate=self.multi_head_fuse)

        # self.TCN=nn.Conv2d(self.GCN_out_dim * self.num_nodes, self.GCN_out_dim * self.num_nodes, kernel_size=(1, self.time_step), stride=(1,1), bias=True)
        # self.TCN = nn.Linear(self.time_step, 1)
        self.TCN = nn.Conv2d(self.GCN_out_dim, self.TCN_out_dim, kernel_size=(1, self.time_step),padding=(0,0),
                          stride=(1,1), bias=True)
        # self.TCN = nn.Conv2d(self.GCN_out_dim, self.TCN_out_dim, kernel_size=(1, 3),padding=(0,1),
        #                   stride=(1,1), bias=True)
        self.gru = GRU(length=self.time_step, in_dim=self.GCN_out_dim, hidden_dim=self.GCN_out_dim * 2, out_dim=self.TCN_out_dim, device=self.device)


        self.TCN_adj = nn.Linear(self.time_step, 1)

        self.LSTM = nn.LSTM(self.num_nodes, self.num_nodes, batch_first=True)  # b*n,l,c


    def forward(self, x):
        #x: b node*t
        # x = self.bn(x)
        if self.time_step == 1:
            x = x.reshape(-1, self.num_nodes, 1, 1)  #（b node d 1）
        elif self.time_step > 1:
            x = x.unsqueeze(-1).reshape(-1, self.num_nodes, 1, self.time_step) #（b node d t）


        if self.start_dim > 1:
            x = F.relu(self.fc0(x)) #(b,node,start_dim) ，由于时间维度扩充，现变为(b,node,start_dim，t)

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None #每次forward，new_supports都被清空了，注意每次forward完(经历一个batch数据，称为iteration)后都接backward更新参数的
        # if self.supports is not None:
        if self.adaptive_GCN:
            if not self.multi_head_fuse:
                self.adp = self.multi_head_A(x if self.adaptive_GCN else None) #(head,792,792),只要list(adp)即可转换为长度为4，元素为(792,792)的矩阵列表
                new_supports = self.supports + list(self.adp)
                # self.adp = torch.concat((self.supports[0].unsqueeze(0), self.adp),dim=0)
            else:
                self.adp, self.adp_fuse = self.multi_head_A(x if self.adaptive_GCN else None)
                new_supports = self.supports + list(self.adp_fuse)
        else:
            new_supports = self.supports
        # x = self.gconv(x, new_supports).reshape(-1, self.start_dim*self.num_nodes)  #(b,node,start_dim)  #reshape (b,node*start_dim)


        if self.GCN_layer < 2:
            # x = F.relu(self.gconv(x, new_supports).reshape(-1, self.GCN_out_dim*self.num_nodes))  #(b,node,GCN_out_dim)  #reshape (b,node*start_dim)
            x = F.relu(self.gconv(x, new_supports))
            if self.time_step == 1:
                x = x.reshape(-1, self.GCN_out_dim * self.num_nodes)  #时间维度是1，直接忽略
            elif self.time_step > 1:
                # x = x.reshape(-1, self.GCN_out_dim * self.num_nodes, self.time_step)
                # x = self.TCN(x).squeeze(-1) #时间维度大于1，先时间卷积提取时间特征，

                x = x.permute(0, 1, 3, 2)
                # x = F.relu(self.TCN(x)).squeeze(-1).reshape(-1, self.GCN_out_dim * self.num_nodes)
                x = F.relu(self.TCN(x)).squeeze(-1).reshape(-1, self.TCN_out_dim * self.num_nodes)
                # x = F.relu(self.TCN(x)).reshape(-1, self.TCN_out_dim * self.num_nodes * self.time_step)
                # x = self.gru(x).squeeze(-1).reshape(-1, self.TCN_out_dim * self.num_nodes)

        else:
            x = F.relu(self.gconv(x, new_supports))
            residual = x
            x = F.relu(self.gconv1(x, new_supports))
            x = (x + residual).reshape(-1, self.GCN_out_dim * self.num_nodes)

        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        if self.hidden_dim_a is not None:
            x = F.relu(self.fc1(x))

        actions = torch.tanh(self.action_out(x))

        if torch.isnan(actions).any(): # 有一个True（非NAN）则都为 True
            print('error nan:', )
        return actions



class CNet_GCN(nn.Module):
    def __init__(self, s_dim, a_dim, cfg, device='cpu'):
        super(CNet_GCN,self).__init__()
        self.s_dim = s_dim
        self.start_dim = cfg.start_dim
        self.hidden_dim = cfg.hidden_dim
        self.GCN_out_dim = cfg.GCN_out_dim
        self.num_nodes = cfg.num_nodes
        self.device = device
        self.adaptive_GCN = cfg.adaptive_GCN
        self.multi_head_fuse = cfg.multi_head_fuse
        self.GCN_layer = cfg.GCN_layer

        # self.GCN_layer = 1
        self.adp = None
        self.adp_fuse = None
        if cfg.supports is False:
            self.supports = []
        else:
            # data_path = r'../data/adjacency_P_self_30.csv' if self.num_nodes == 30 else r'../data/adjacency_P_self_57.csv'
            data_path = r'../data/adjacency_P_self_{}.csv'.format(self.num_nodes)
            self.supports = [torch.tensor(load_adj(data_path), requires_grad=False).float().to(device)]

        if self.multi_head_fuse:
            head_num_multigcn = 1
        else:
            head_num_multigcn = cfg.head_num
        # supports_len = 1 if (cfg.supports is False or cfg.adaptive_GCN is False) else 1+cfg.head_num
        supports_len = 1 if (cfg.supports is True and cfg.adaptive_GCN is False) \
            else (1 + head_num_multigcn) if (cfg.supports is True and cfg.adaptive_GCN is True) \
            else head_num_multigcn
        # if (cfg.supports is False and cfg.adaptive_GCN is True)

        if self.dynamic_GCN is True:
            supports_len += 1


        self.time_step = cfg.time_step
        self.TCN_out_dim = cfg.TCN_out_dim

        # self.fc0s = nn.Linear(1, self.start_dim)
        self.fc0s = nn.Conv2d(1, self.start_dim, kernel_size=(1, 1), stride=(1, 1), bias=True)  # 4维向量扩充
        # self.fc0a = nn.Linear(a_dim, self.start_dim)
        # self.fcsa = nn.Linear(self.start_dim*s_dim+self.start_dim, 64)

        self.GCN_model = cfg.GCN_model
        if self.GCN_model == 'DGCN':
            # self.gconv = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)(self.start_dim, self.GCN_out_dim, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x
            self.gconv = cheby_conv_ds(self.start_dim, self.GCN_out_dim, K=2, device=self.device) #这个K就是multi_gcn里的order+1
        elif self.GCN_model == 'GWNET':
            self.gconv = multi_gcn(self.start_dim, self.GCN_out_dim, order=1, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x


        # self.gconv1 = multi_gcn(self.GCN_out_dim, self.GCN_out_dim, support_len=supports_len, dynamic_GCN=self.dynamic_GCN)  # dropout None or 0.x
        self.gconv1 = None
        if self.hidden_dim is not None:
            # self.fcsa = nn.Linear(self.GCN_out_dim*s_dim+a_dim, self.hidden_dim)
            # self.fcsa = nn.Linear(self.GCN_out_dim*self.num_nodes+a_dim, self.hidden_dim)
            self.fcsa = nn.Linear(self.TCN_out_dim*self.num_nodes+a_dim, self.hidden_dim)
            # self.fcsa = nn.Linear(self.time_step*self.TCN_out_dim*self.num_nodes+a_dim, self.hidden_dim)
            self.q_out = nn.Linear(self.hidden_dim, 1)
        else:
            self.q_out = nn.Linear(self.GCN_out_dim*self.num_nodes+a_dim, 1)


        self.multi_head_A = multi_head_A(head=cfg.head_num, node_embedding=cfg.node_embedding, num_nodes=self.num_nodes, device=device, epsilon=cfg.epsilon, aggregate=self.multi_head_fuse)

        # self.TCN=nn.Conv2d(self.GCN_out_dim * self.num_nodes, self.GCN_out_dim * self.num_nodes, kernel_size=(1, self.time_step), stride=(1,1), bias=True)
        # self.TCN = nn.Linear(self.time_step, 1)
        self.TCN = nn.Conv2d(self.GCN_out_dim, self.TCN_out_dim, kernel_size=(1, self.time_step),padding=(0,0),
                          stride=(1,1), bias=True)
        # self.TCN = nn.Conv2d(self.GCN_out_dim, self.TCN_out_dim, kernel_size=(1, 3),padding=(0,1),
        #                   stride=(1,1), bias=True)
        self.gru = GRU(length=self.time_step, in_dim=self.GCN_out_dim, hidden_dim=self.GCN_out_dim * 2, out_dim=self.TCN_out_dim, device=self.device)

        self.TCN_adj = nn.Linear(self.time_step, 1)

        self.LSTM = nn.LSTM(self.num_nodes, self.num_nodes, batch_first=True)  # b*n,l,c

    def forward(self, s, a):
        # x = torch.cat([s, a], dim=1)
        # x = F.relu(self.fc1(x))

        # s = s.unsqueeze(-1) ##unsqueeze(-1)将输入向量[30]，变为一个节点一个特征的图输入数据[30,1]。
        #x: b node*t
        # x = self.bn(x)
        if self.time_step == 1:
            s = s.reshape(-1, self.num_nodes, 1, 1)  #（b node d 1）
        elif self.time_step > 1:
            s = s.unsqueeze(-1).reshape(-1, self.num_nodes, 1, self.time_step) #（b node d t）


        if self.start_dim > 1:
            s = F.relu(self.fc0s(s)) #(b,node,start_dim)

        # a = F.relu(self.fc0a(a)) #仅有自己的action，不是图数据，所以直接线性映射

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None #每次forward，new_supports都被清空了，注意每次forward完(经历一个batch数据，称为iteration)后都接backward更新参数的

        if self.adaptive_GCN:
            if not self.multi_head_fuse:
                self.adp = self.multi_head_A(s if self.adaptive_GCN else None) #(head,792,792),只要list(adp)即可转换为长度为4，元素为(792,792)的矩阵列表
                new_supports = self.supports + list(self.adp)
                # self.adp = torch.concat((self.supports[0].unsqueeze(0), self.adp), dim=0)
            else:
                self.adp, self.adp_fuse = self.multi_head_A(s if self.adaptive_GCN else None)
                new_supports = self.supports + list(self.adp_fuse)
        else:
            new_supports = self.supports


        if self.GCN_layer < 2:
            # x = F.relu(self.gconv(x, new_supports).reshape(-1, self.GCN_out_dim*self.num_nodes))  #(b,node,GCN_out_dim)  #reshape (b,node*start_dim)
            s = F.relu(self.gconv(s, new_supports))
            if self.time_step == 1:
                s = s.reshape(-1, self.GCN_out_dim * self.num_nodes)  #时间维度是1，直接忽略
            elif self.time_step > 1:
                # s = s.reshape(-1, self.GCN_out_dim * self.num_nodes, self.time_step)
                # s = self.TCN(s).squeeze(-1) #时间维度大于1，先时间卷积提取时间特征，

                s = s.permute(0, 1, 3, 2)
                # s = F.relu(self.TCN(s)).squeeze(-1).reshape(-1, self.GCN_out_dim * self.num_nodes)
                s = F.relu(self.TCN(s)).squeeze(-1).reshape(-1, self.TCN_out_dim * self.num_nodes)
                # s = F.relu(self.TCN(s)).reshape(-1, self.TCN_out_dim * self.num_nodes * self.time_step)
                # s = self.gru(s).squeeze(-1).reshape(-1, self.TCN_out_dim * self.num_nodes)

        else: #还没改
            x = F.relu(self.gconv(x, new_supports))
            residual = x
            x = F.relu(self.gconv1(x, new_supports))
            x = (x + residual).reshape(-1, self.GCN_out_dim * self.num_nodes)


        x = torch.cat([s, a], dim=1)
        if self.hidden_dim is not None:
            x = F.relu(self.fcsa(x))
        actions_value = self.q_out(x)

        return actions_value






