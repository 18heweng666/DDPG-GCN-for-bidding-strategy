function [lam,quantity,price,earnings,total_load_percentage,success,f] = rl_auction_4bus_disp(action, load, mpc, verbose)
%%
mkt.OPF = 'DC'; 
mkt.auction_type = 1;
mpopt = mpoption;
mpopt.out.all = 0;
if verbose
%     mpopt.out.all = 1; #两句都行
    mpopt = mpoption;
end

% mpopt.out.sys_sum = 0;
% mpopt.out.bus = 0;
% mpopt.out.branch = 0;
% mpopt.out.lim.all = 0;
%%
% 6个发电商，3个售电商（可调度负荷），17个固定负荷fixed load，9个个体参与竞价，其他load作为固定值存储在bus的Pd中
% if offers not defined, use gencost
% mpc = loadcase('case4_disp');

n_agent = 8;

% mpc.gen(1:4,9) = [200; 250; 300; 300]; 
mpc.gen(5:n_agent,10) = -load; %time-varying load %python输入过来的都是正数

% offers.P.qty = [200; 250; 300; 300]; %chat:6个发电商提交了报价报量,行为发电商，列表示容量档位，元素表示愿意出售的最大电力量   DOC:Six generators with three blocks of capacity
% offers.P.prc = [23; 30; 24.5; 24.5 ]; 
% bids.P.qty = [100; 200; 120; 320]; %3个售电商提交了报价报量，同理 
% bids.P.qty = [100; 200; 220; 320];
% bids.P.prc = [33; 20; 16; 25]; %Three dispatchable loads, bidding three blocks
offers.P.qty = mpc.gen(1:4,9) .* action(n_agent+1:n_agent+4)';  %9是最大发电量所在列
bids.P.qty = abs(mpc.gen(5:n_agent,10)) .* action(n_agent+5:end)'; 

%action行向量
% action = [1;1;1;1;1;1;1;1]';
% action = [1.5;1.5;1.5;1.5;0.9;1;0.9;1]';
% action = [1.5;1.5;1.5;1.5;0.9;0.9;0.9;0.9]';
% offers.P.prc = action(1:4)'.*( 2.*mpc.gencost(1:4,5).*offers.P.qty + mpc.gencost(1:4,6) );
% bids.P.prc = action(5:n_agent)'.*(2.*mpc.gencost(5:n_agent,5).*bids.P.qty + mpc.gencost(5:n_agent,6));

offers.P.prc = action(1:4)'.*( 2.*mpc.gencost(1:4,5).*offers.P.qty + mpc.gencost(1:4,6) );
bids.P.prc = action(5:n_agent)'.*(-2.*mpc.gencost(5:n_agent,5).*bids.P.qty + mpc.gencost(5:n_agent,6));
    


%%
[mpc_out, co, cb, f, dispatch, success, et] = runmarket(mpc, offers, bids, mkt, mpopt);
% [mpc_out, co, cb, f, dispatch, success, et] = runmarket(mpc, [], [], mkt);
% [mpc_out, co, cb, f, dispatch, success, et] = runmarket(mpc, offers, bids, mkt);
lam = mpc_out.bus(:,14); %所有节点的电价
% genco = co.P; 
% load = cb.P;

[QUANTITY, PRICE, FCOST, VCOST, SCOST, PENALTY] = idx_disp; %dispatch矩阵的各个列意义
quantity = abs(dispatch(:,QUANTITY)); %包括了发电商和售电商的清算电量，前面是发电商，后面是售电商
price= dispatch(:,PRICE); %%包括了发电商和售电商所在节点的清算电价

t=1; % the default duration t is 1 hour
pay = dispatch(:, PRICE) .* dispatch(:, QUANTITY) * t; 
cost = dispatch(:, FCOST) + dispatch(:, VCOST) + dispatch(:, SCOST) + dispatch(:, PENALTY);
earnings = pay-cost;

% 获取线路的功率信息和额定功率
pf = mpc_out.branch(:,14);
% pf = pf ./ mpc_out.baseMVA;
% pt = mpc_out.branch(:,16);
rate_A = mpc_out.branch(:, 6);


% 计算线路的负载情况
% load_percentage_positive = abs(pf ./ rate_A);
% load_percentage_negative = abs(pt ./ rate_A);
% total_load_percentage = load_percentage_positive + load_percentage_negative;
% total_load_percentage = load_percentage_positive .* 100;

total_load_percentage = abs(pf);


end