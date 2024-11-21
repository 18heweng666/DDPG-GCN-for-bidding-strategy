function [lam,quantity,price,earnings,total_load_percentage,success,f] = rl_auction_30bus_disp_self(action, load, mpc, verbose)
%%
% verbose = 1;
mkt.OPF = 'DC'; 
mkt.auction_type = 1;
mpopt = mpoption;
mpopt.out.all = 0;
if verbose
%     mpopt.out.all = 1; #两句都行
    mpopt = mpoption;
end
% mpopt.out.all = 1;

% mpopt.out.sys_sum = 0;
% mpopt.out.bus = 0;
% mpopt.out.branch = 0;
% mpopt.out.lim.all = 0;
%%
% 6个发电商，3个售电商（可调度负荷），17个固定负荷fixed load，9个个体参与竞价，其他load作为固定值存储在bus的Pd中
% if offers not defined, use gencost
% mpc = loadcase('case30_disp_self');
% mpc = loadcase('t_auction_case');

n_agent = 9;

% mpc.gen(1:6,9) = [30; 60; 150; 100; 160; 60]; 
mpc.gen(7:n_agent,10) = -load; %%time-varying load

% offers.P.qty = [60; 60; 60; 60; 60; 60]; %chat:6个发电商提交了报价报量,行为发电商，列表示容量档位，元素表示愿意出售的最大电力量   DOC:Six generators with three blocks of capacity
% offers.P.qty = [30; 60; 150; 100; 160; 60]; 
% offers.P.prc = [23; 30; 24.5; 24.5 ]; 
% bids.P.qty = [30; 30; 30]; %3个售电商提交了报价报量，同理 
% bids.P.qty = [80; 120; 100];
% bids.P.qty = [100; 200; 220; 320];
% bids.P.prc = [33; 20; 16; 25]; %Three dispatchable loads, bidding three blocks
offers.P.qty = mpc.gen(1:6,9) .* action(n_agent+1:n_agent+6)';  %9是最大发电量所在列
bids.P.qty = abs(mpc.gen(7:n_agent,10)) .* action(n_agent+7:end)'; 
% disp(bids.P.qty)
%action行向量
% action = [1;1;1;1;1;1;1;1;1]';
% action = [1.5;1.5;1.5;1.5;0.9;0.9;0.9;0.9]';
offers.P.prc = action(1:6)'.*( 2.*mpc.gencost(1:6,5).*offers.P.qty + mpc.gencost(1:6,6) );
bids.P.prc = action(7:n_agent)'.*(-2.*mpc.gencost(7:n_agent,5).*bids.P.qty + mpc.gencost(7:n_agent,6) );
% disp(bids.P.prc)


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
% total_load_percentage = load_percentage_positive .* 100;

total_load_percentage = abs(pf);


end