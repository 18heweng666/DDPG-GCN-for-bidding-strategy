function mpc = case4_disp
%CASE5  Power flow data for modified 5 bus, 5 gen case based on PJM 5-bus system
%   Please see CASEFORMAT for details on the case file format.
%
%   Based on data from ...
%     F.Li and R.Bo, "Small Test Systems for Power System Economic Studies",
%     Proceedings of the 2010 IEEE Power & Energy Society General Meeting

%   Created by Rui Bo in 2006, modified in 2010, 2014.
%   Distributed with permission.

%   MATPOWER

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	2	0	0	0	0	1	1	1	230	1	1.1	0.9;
	2	2	0	0	0	0	1	1	0	230	1	1.1	0.9;
	3	2	0	0	0	0	1	1	0	230	1	1.1	0.9;
    4	3	0	0	0	0	1	1	0	230	1	1.1	0.9;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	100	0	0	0	1	100	1	200	0;
	2	100	0	0	0	1	100	1	250	0;
    3	100	0	0	0	1	100	1	300	0;
    4	100	0	0	0	1	100	1	300	0;
    
	1	-50	0	0	0	1	100	1	0	-100;
	2	-50	0	0	0	1	100	1	0   -200;
    3	-50	0	0	0	1	100	1	0   -120;
%     3	-50	0	0	0	1	100	1	0   -220;
    4	-50	0	0	0	1	100	1	0   -320;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
% mpc.branch = [
%     1	2	0.00281	0.0281	0.00712	0	0	0	0	0	1	-360	360; %uncongensted
%     2	3	0.00064	0.0064	0.03126	0	0	0	0	0	1	-360	360;
%     3	4	0.00304	0.0304	0.00658	0	0	0	0	0	1	-360	360;
%     1	4	0.00064	0.0064	0.03126	0	0	0	0	0	1	-360	360;
% ];
mpc.branch = [    
    1	2	0.0281	0.0281	0.0712	50	50	50	0	0	1	-360	360; %设置了branch flow limits：rateA	rateB	rateC，DCOPF才会有节点电价，否则统一价格
    2	3	0.00064	0.0064	0.03126	0	0	0	0	0	1	-360	360;
    3	4	0.00304	0.0304	0.00658	0	0	0	0	0	1	-360	360;
    1	4	0.064	0.064	0.03126	50	50	50	0	0	1	-360	360;
];
% mpc.branch = [    
%     1	2	0.0281	0.0281	0.0712	150	150	150	0	0	1	-360	360; %设置了branch flow limits：rateA	rateB	rateC，DCOPF才会有节点电价，否则统一价格
%     2	3	0.00064	0.0064	0.03126	0	0	0	0	0	1	-360	360;
%     3	4	0.00304	0.0304	0.00658	200	200	200	0	0	1	-360	360;
%     1	4	0.064	0.064	0.03126	150	150	150	0	0	1	-360	360;
% ];

%在matpower中，mpc.branch中的RATE A是指在长期情况下分配给支路的最大容量限制，以兆伏安（MVA）为单位。如果RATE A为0，则表示该支路没有容量限制。RATE A*越大表示支路的容量限制越高，即支路的最大容量限制更大。
%所以是越小（非0），越容易导致线路阻塞
%这样设置会无法收敛
%	1	2	0	0	0	0	0	0	0	0	1	-360	360;
% 	2	3	0	0	0	0	0	0	0	0	1	-360	360;
% 	3	4	0	0	0	0	0	0	0	0	1	-360	360;
%   1	4	0	0	0	0	0	0	0	0	1	-360	360;
%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0    Quadratic"成本模型
% mpc.gencost = [
% 	2	0	0	3	0.025   18	0;
% 	2	0	0	3	0.025	20  0;
% 	2	0	0	3	0.035   21	0; %原始的，genco3成本太高，难成交
% 	2	0	0	3	0.025	17  0;
%     
% 	2	0	0	3   -0.05	60	0;
% 	2	0	0	3   -0.05	70	0;
%     2	0	0	3   -0.05	80	0;
%     2	0	0	3   -0.05	100	0;
% ];

mpc.gencost = [
% 	2	0	0	3	0.028   19	0;
% 	2	0	0	3	0.025	20  0;
% 	2	0	0	3	0.032   22	0;  %这样设有助于genco3有成交
% 	2	0	0	3	0.025	17  0;
    
    2	0	0	3	0.025   18	0;
	2	0	0	3	0.025	20  0;
	2	0	0	3	0.035   21	0;  %原文
	2	0	0	3	0.025	17  0;
    
% 	2	0	0	3   -0.05	60	0;
% 	2	0	0	3   -0.05	70	0;
%     2	0	0	3   -0.05	80	0;
%     2	0	0	3   -0.05	100	0;
    
    2	0	0	3   0.05	60	0;
	2	0	0	3   0.05	70	0;
    2	0	0	3   0.05	80	0;
    2	0	0	3   0.05	100	0;

];