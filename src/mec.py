import numpy as np
import math
import argparse

'''
要实现的方法：

num_actions = env.action_space.shape[0]
num_states = env.observation_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


prev_state = env.reset()
state, reward, done, info = env.step(action)

'''

def proper_edge_loc(edge_num, user_num):
    # initial the e_l
    e_l = np.zeros((edge_num, 2))
    # calculate the mean of the data
    # group_num = math.floor(user_num / edge_num) # 向下取整
    group_num = round(user_num / edge_num) # 每个边缘服务器
    edge_id = 0
    for base in range(0, group_num*edge_num, group_num):
        for data_num in range(base, base + group_num):
            data_name = str("%03d" % (data_num + 1))  # plus zero
            file_name = "KAIST" + "_30sec_" + data_name + ".txt"
            file_path = "data/" + "KAIST" + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            # get line_num and initial data
            line_num = 0
            for line in f1:
                line_num += 1
            data = np.zeros((line_num, 2))
            # collect the data from the .txt
            index = 0
            for line in f1:
                data[index][0] = line.split()[1]  # x
                data[index][1] = line.split()[2]  # y
                index += 1
            # stack the collected data
            if data_num % group_num == 0:
                cal = data
            else:
                cal = np.vstack((cal, data))
        e_l[edge_id] = np.mean(cal, axis=0)
        edge_id += 1
    return e_l


TXT_NUM = 92
LOCATION = "KAIST"
def get_minimum():
    cal = np.zeros((1, 2))
    for data_num in range(TXT_NUM):
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_name + ".txt"
        file_path = "data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num
        line_num = 0
        for line in f1:
            line_num += 1
        # collect the data from the .txt
        data = np.zeros((line_num, 2))
        index = 0
        for line in f1:
            data[index][0] = line.split()[1]  # x
            data[index][1] = line.split()[2]  # y
            index += 1
        # put data into the cal
        cal = np.vstack((cal, data))
    return min(cal[:, 0]), min(cal[:, 1])

# 先固定终端用户的发射功率
class Env():
    # 初始化系统
    def __init__(self, B, USER_NUM, EDGE_NUM, F, f, Dn, Cn, pn, pi, w1 = 0.5, w2 = 0.5):
        # B 带宽 10 MHz
        # USER_NUM 用户数量
        # EDGE_NUM 边缘服务器数量
        # F 边缘服务器总计算能力 GHz/sec (1000*1000*1000)
        # f 用户计算能力        GHz/sec (1000*1000*1000)
        # Dn, Cn 任务量大小，所需cpu周期数, (300~500kbits), (900, 1100)兆周期数 1Mhz = 1000khz = 1000*1000hz
        # pn, pi 上传功率，闲时功率 | mW (毫瓦)
        # state 系统状态

        #####################  hyper parameters  ####################
        LOCATION = "KAIST"
        self.USER_NUM = USER_NUM  # 用户数量
        self.EDGE_NUM = EDGE_NUM  # 边缘服务数量
        self.B, self.F = B, F
        self.pn, self.pi = pn, pi
        self.Dn, self.Cn, self.f = Dn, Cn, f

        self.w1, self.w2 = w1, w2
        print("self w1, self w2 ", self.w1, self.w2)

        MAX_EP_STEPS = 3000  # 一个epoch最大的步数
        TXT_NUM = 92

        self.state = 0
        self.reward = 0

        self.user = {}
        #####################  初始化用户（字典从1开始）  ####################
        for data_num in range(self.USER_NUM):
            # 添加轨迹
            data_num = str("%03d" % (data_num + 1))  # plus zero
            file_name = LOCATION + "_30sec_" + data_num + ".txt"
            file_path = "data/" + LOCATION + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            data = 0
            for line in f1:
                data += 1
            num_step = data * 30
            mob = np.zeros((num_step, 2))

            # write data to self.mob
            now_sec = 0
            for line in f1:
                for sec in range(30):
                    mob[now_sec + sec][0] = line.split()[1]  # x
                    mob[now_sec + sec][1] = line.split()[2]  # y
                now_sec += 30
            self.user["user"+str(data_num)] = mob

        #####################  初始化边缘服务器（字典从1开始）  ####################
        e_l = proper_edge_loc(self.EDGE_NUM, self.USER_NUM)
        self.edge = {}
        for i in range(self.EDGE_NUM):
            self.edge["edge"+str("%03d"%(i+1))+"loc"] = e_l[i, :]


    def get_pos(self, render):
        loc = np.zeros((self.USER_NUM, 2))
        for i in range(self.USER_NUM):
            loc[i][0] = self.user["user" + str("%03d" % (i + 1))][render][0]
            loc[i][1] = self.user["user" + str("%03d" % (i + 1))][render][1]
        return loc

    # 在render时间时，采取action的动作，返回延迟和能量以及用户在render时的位置
    # render = 1(标量)
    # 要求输入进来的action, action_p, action_f 必须已经合法
    def step(self, render, action, action_p, action_f):
        # 返回用户位置
        # loc = np.zeros((self.USER_NUM, 2))
        # for i in range(self.USER_NUM):
        #     loc[i][0] = self.user["user"+str("%03d" % (i+1))][render][0]
        #     loc[i][1] = self.user["user"+str("%03d" % (i+1))][render][1]
        loc = self.get_pos(render)

        # action.shape   = (n, m+1)  对应每个任务的计算卸载决策：0 本地， 1, ... , m
        # action_p.shape = (n, )     对应每个用户的发射功率的调整
        # action_f.shape = (m, n)    对应每个边缘服务器分配资源量
        action = action.reshape((self.USER_NUM, self.EDGE_NUM+1))
        action_p = action_p.reshape((self.USER_NUM, ))
        action_f = action_f.reshape((self.EDGE_NUM, self.USER_NUM))

        action = np.argmax(action, 1)

        delay = 0
        energy = 0

        # x dbm = y mW
        # x/10 = log10(y)

        mw = pow(10, -174 / 10) * 0.001 # 此时单位为W
        for i in range(self.USER_NUM):
            # local
            if action[i] == 0:
                tl1 = self.Cn[i] / (self.f * 1000)
                el1 = self.f * self.f * 0.001 * self.Cn[i]
                delay += tl1
                energy+= el1
            # offloading
            else:
                # rn = trans_rate(loc[i], self.edge["edge"+str("%03d"%action[i])]["loc"])
                u_e_dist = np.sqrt(np.sum(np.square(loc[i] - self.edge["edge"+str("%03d"%(action[i]))+"loc"]))) + 0.01
                # print("dist is ", u_e_dist)
                # rn = self.B * 1000 * 1000 * np.log2(1 + self.pn * 0.001 * np.power(u_e_dist, -3) / mw)
                rn = self.B * 1000 * 1000 * np.log2(1 + action_p[i] * 0.001 * np.power(u_e_dist, -3) / mw)
                # rn = self.B / self.USER_NUM * 1000 * 1000 * np.log2(1 + action_p[i] * 0.001 * np.power(u_e_dist, -3) / (self.B / self.USER_NUM * 1000 * 1000 * mw))
                # 卸载上传时间延迟
                to1 = self.Dn[i] * 1024 / rn
                # 卸载至服务器计算延迟
                # to2 = self.Cn[i] / (self.F * 1000 / self.USER_NUM)  # 平均分配资源
                to2 = self.Cn[i] / (action_f[action[i]-1][i] * 1000)

                # 能量消耗
                # eo1 = to1 * self.pn * 0.001
                # eo2 = to2 * self.pi * 0.001
                eo1 = to1 * action_p[i] * 0.001
                eo2 = to2 * self.pi * 0.001

                delay += (to1 + to2)
                energy +=(eo1 + eo2)
        # print("delay = ", delay)
        # print("energy = ", energy)
        cost = delay * self.w1 + energy * self.w2
        # cost = delay * 0.5 + energy * 0.5
        # print("cost = ", cost)
        return loc, delay, energy, cost

    def reset(self, user_num, edge_num):
        action = np.zeros((user_num, edge_num + 1))
        action[:, 0] = 1
        loc, delay, energy, cost = self.step(0, action=action, action_p=np.array([500] * user_num), action_f=np.zeros((edge_num, user_num)))
        return loc, delay, energy, cost

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ue', type=int, default=5)
    parser.add_argument('--F', type=int, default=5)
    args = parser.parse_args()
    return args
