import numpy as np
import matlab.engine
import matlab  #matlab用于数据转换等工作
from run.utils import *
# eng.doc(nargout=0)   # 将直接打开浏览器MATLAB文档界面
# 一维数组
# a = matlab.double([9,16])  # list或 a = matlab.double((9,16)) tuple
# 多维数组
# a = matlab.double([[1,2,3,4],[5.0,6.0,7.0,8.0]])
# print(a)           # 输出：[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]
# print(a.size)      #属性 输出：(2,4)
# print(a[0][1])     #索引 输出：2.0
# print(a[0][1:3])   #切片 输出：[2.0,3.0]
# print(a.reshape((4,2)))   #重构 输出：[[1.0,3.0],[5.0,7.0],[2.0,4.0],[6.0,8.0]]

# Python 索引是从 0 开始的。当在 Python 会话中访问 MATLAB 数组的元素时，请使用从 0 开始的索引.

#如果 MATLAB 函数不在路径中，您可以从当前文件夹中调用它。例如，要调用文件夹 myFolder 中的 MATLAB 函数 myFnc，请键入：
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.cd(r'myFolder', nargout=0)
# eng.myFnc()





# print('d', d, type(d))

# eng.quit()
# 市场出清，考虑网络阻塞
def market_clearing(a_declare, load, eng, mpc, case_name='case4_disp', verbose=True, wind=None):
    # eng = matlab.engine.start_matlab()  # 可以调用matlab的内置函数。
    # d = eng.smartmarket # 可以调用matlab写的脚本函数
    # d = eng.max( matlab.double(np.array([3,1,5])) ) #python list 要转换为matlab double来输入
    # mpc_out, co, cb, f, dispatch, success, et = eng.rl_auction_30bus  #python list 要转换为matlab double来输入

    # lam,quantity,price_forcasting,earnings = eng.rl_auction_30bus(nargout=4)  #  当使用引擎调用函数时，默认情况下该引擎会返回单个输出参数。如果您知道函数可能返回多个参数，请使用 nargout 参数指定输出参数的数量。
    # lam = [i[0] for i in list(lam)]  #list和double的类型可以互相转换
    # quantity = [i[0] for i in list(quantity)]
    # price_forcasting = [i[0] for i in list(price_forcasting)]
    # earnings = [i[0] for i in list(earnings)]
    if case_name == 'case4_disp':
        temp = eng.rl_auction_4bus_disp(matlab.double(list(a_declare)), matlab.double(list(load)), mpc, verbose, nargout=7)  # 输出tuple：4  注意从python传给matlab的数据最好不要是int，要是float！因为int和matlab的double不相容
    elif case_name == 'case30_disp_self':
        temp = eng.rl_auction_30bus_disp_self(matlab.double(list(a_declare)), matlab.double(list(load)), mpc, verbose, nargout=7)  # 输出tuple：4  注意从python传给matlab的数据最好不要是int，要是float！因为int和matlab的double不相容
    elif case_name == 'case30_disp_wind_self':
        temp = eng.rl_auction_30bus_disp_wind_self(matlab.double(list(a_declare)), matlab.double(list(load)), matlab.double(list(wind)), mpc, verbose, nargout=7)  # 输出tuple：4  注意从python传给matlab的数据最好不要是int，要是float！因为int和matlab的double不相容
    elif case_name == 'case57_disp_wind_self':
        temp = eng.rl_auction_57bus_disp_wind_self(matlab.double(list(a_declare)), matlab.double(list(load)), matlab.double(list(wind)), mpc, verbose, nargout=7)  # 输出tuple：4  注意从python传给matlab的数据最好不要是int，要是float！因为int和matlab的double不相容
    else:
        temp = eng.rl_auction_57bus_disp_self(matlab.double(list(a_declare)), 0, mpc, verbose,nargout=7)  # 输出tuple：4  注意从python传给matlab的数据最好不要是int，要是float！因为int和matlab的double不相容
    result_key = ['lam', 'quantity', 'price_forcasting', 'earnings', 'total_load_percentage', 'success', 'f']
    result = {}
    for j in range(len(temp)-2):
        result[result_key[j]] = [i[0] for i in list(temp[j])]
    result[result_key[-2]] = temp[-2]
    result[result_key[-1]] = temp[-1]

    if not result['success']:
        print('not converge action:' + str(a_declare))

    return result['price_forcasting'], result['earnings'], result['f'], result['quantity'], -result['f'], result['lam'], result['total_load_percentage']

if __name__ == '__main__':

    case_name = 'case30_disp_self'  # case30_disp_self case4_disp

    n_agents = 8 if case_name == 'case4_disp' else 9 if case_name == 'case30_disp_self' else 10
    n_agents_gen = 4 if case_name == 'case4_disp' else 6 if case_name == 'case30_disp_self' else 7

    alpha = np.array([21.388, 23.807, 34.317, 27.235, 33.609, 24.848, 23.3, 35])
    alpha = np.array([1.5, 1.6, 1, 1.5, 1.0, 1, 1.8,  1, 1, 1])

    # alpha = np.array([1.435, 0.97,  1.045, 1.16,  0.85,  0.885, 0.99,  0.9 ])
    # alpha = np.array([1.435, 0.97,  1.045, 1.16,  0.85,  0.885, 0.99,  0.95  ])
    # alpha = np.array([1.42,  1.23,  1.,    0.96,  0.89,  0.855, 0.875, 0.925 ])
    # alpha = np.array([1.435, 1.13,  1.015, 1.15,  0.985, 0.9,   0.87,  0.945] )

    # alpha = np.array([1.323108609, 0.900020626,  0.900026492, 0.90008806,  0.999906307, 0.999978981,   0.880038917,  0.999993093] )
    # alpha = np.array([1.328, 0.900, 0.900, 0.900, 0.999, 0.999, 0.999, 0.999])
    # alpha = np.array([1.328, 0.900, 1.1500, 0.900,  0.999, 0.999, 0.999, 0.999])
    # alpha = np.array([1.328, 0.900, 0.900, 2.000,  0.999, 0.999, 0.999, 0.999])
    # alpha = np.array([1.7, 1.500, 1.1500, 0.900,  0.999, 0.999, 0.999, 0.999])


    alpha = np.array([1.3167627073824406,0.9000057697296143,0.9000462234020233,0.9001909255981445,0.9999264359474183,0.9999893829226494,0.8501561984419822,0.9999995797872543])










    #30 bus MADDPG-(PER) nash
    # alpha = np.array([0.94983245, 0.9,        0.9,        0.93206714, 0.9,        0.90289919, 0.99268068, 0.85,       0.99793857])
    # alpha = np.array([0.9703413,  0.91601221, 0.95526187, 0.94047227, 0.9,        0.9,0.97399644, 0.85, 1.])
    alpha = np.array([0.9850253, 0.9,        0.97797693, 0.94642933, 0.9,        0.95398392, 1.,         0.85,       0.99274116]) #good
    # alpha = np.array([0.900057238,	0.900031078,	0.900158077,	0.900014424,	0.900145522,	0.90038949,	0.99998789,	0.850022092,	0.99995594]) #RL收敛点

    #对角化 30bus
    # alpha = np.array([0.92931257, 0.91218848, 0.90933488, 0.9221494,  0.9,        0.90011725,1.,         0.85,       1.]) #precision low
    # alpha = np.array([0.9, 0.9, 0.993975482963541, 1.3851839546501283, 1.392174697338488, 1.0877248678338924, 0.85, 0.85, 0.8755062546131753]) #ok 1.019
    # alpha = np.array([0.9464712935823434, 0.9, 0.9809869111217749, 0.9324502195158074, 0.9, 0.9, 1.0, 0.85, 0.8910485951832214]) #low
    # alpha = np.array([0.9, 0.9, 0.9158418477556801, 0.9331540556691161, 1.0671341607198204, 0.9, 0.85, 0.85, 0.9318272574832589]) #ok 1.019
    # alpha = np.array([0.9, 0.9, 0.9, 0.9378842772790678, 1.0825327636819841, 0.9, 0.85, 0.85, 1.0])

    #对角化 4bus，无nash
    # alpha = np.array([1.5102288567055564, 0.9291615029824543, 1.006897293228832, 0.9, 1.0, 0.8968361881014486, 0.85, 0.903932698366469])

    #特意改变
    alpha = np.array([0.90, 0.9, 0.97797693, 0.94642933, 0.9, 0.95398392, 1., 0.85, 0.99274116])  # good

    beta = np.ones_like(alpha)
    market_input = np.concatenate((alpha, beta))
    eng = matlab.engine.start_matlab()

    mpc = eng.loadcase(case_name)
    load_data_default = [100, 200, 120, 320] if case_name == 'case4_disp' else [80, 120, 100] if case_name == 'case30_disp_self' else None
    # load_data_default = [80, 120, 100]
    # [1.02649676 0.9        1.25917934 1.13438089 0.92868374 0.94604912
    #  0.9        0.96190512]
    nodal_price, profit, obj_value, quant, f, _, total_load_percentage = market_clearing(market_input, load_data_default, eng, mpc, case_name=case_name)
    print(np.mean(profit)/1000)

    result_actions, result, result_optimal_index, profit = plot_experiment_nash(market_input, eng, mpc,
                                                                                load_data_default, step_num=[30, 15],
                                                                                case_name=case_name, gen_num=n_agents_gen)

    nash_flag = judge_nash(result_actions, result, result_optimal_index)
    if nash_flag.sum() == len(nash_flag):
        print('nash achieve!!!')
        print('nash action set: ', list(np.round(alpha,2)) )
        print('nash agent average profit: ', np.round(np.mean(profit) / 1000, 4) )
        print('nash agent profit: ', list(np.round(np.array(profit) / 1000, 2)) )
        print('-' * 20)
    else:
        print(nash_flag)

    # plot_nash(result_actions, result, result_optimal_index, gen_num=n_agents_gen)



    plot_agent_list = [0,2,3, 6,7]
    plot_nash(result_actions, result, result_optimal_index, gen_num=3, plot_agent_list=plot_agent_list, subplot=[5,1], figsize=(4,9))
    # print(price)
    # print(earnings