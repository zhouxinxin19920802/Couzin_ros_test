# -*- coding: utf-8 -*-
# @Author  : zhouxin
# @Time    : 2023/12/23 19:35
# @File    : couzin_env_xu.py.py
# @annotation    :
# 日志模块

# 添加 w 自适应功能
# 添加邻居选择功能

import logging
import math
import random
from math import *
import rospy
import copy 
import os
#linux  
import ros
# 信息列表
from couzin_test.msg import agent
from couzin_test.msg import agents


# import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *

logging.basicConfig(
    level=logging.INFO,  # 控制台打印的日志级别
    filename="test_log_0.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

class Field:
    def __init__(self):
        self.width = 500
        self.height = 500


field = Field()


# 将列表转化为列表
def convert_list(obs):
    state = []
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            for k in range(len(obs[i][j])):
                state.append(obs[i][j][k])
    return state



# n * (n - 1) * 4
def convert_list1(obs):
    state = []
    for i in range(len(obs)):
        state_s = []
        for j in range(len(obs[i])):
            state_s = np.concatenate([state_s, obs[i][j]])
        state = np.concatenate([state, state_s])
    return state

# n * (n - 1) * 4
def convert_list3(obs):
    state = []
    for i in range(len(obs)):
        state = np.concatenate([state, obs[i]])
    return state


# n个元素 每个元素(n-1) * 4
def convert_list2(obs):
    state = []
    for i in range(len(obs)):
        state_s = []
        for j in range(len(obs[i])):
            state_s = np.concatenate([state_s, obs[i][j]])
        state.append(state_s)
    return state

# 空间复杂度计算
def cal_space_complexity(swarm):
    sparial_complexity = 0
    for i in range(len(swarm)):
        for j in range(i+1,len(swarm)):
            # 计算两个智能体之间的速度方向的夹角
            agent_i = swarm[i]
            agent_j = swarm[j]
            sparial_complexity = sparial_complexity + cal_angle_of_vector1([agent_i.vel[0], agent_i.vel[1]],[agent_j.vel[0], agent_j.vel[1]])
    sparial_complexity =  sparial_complexity

    return  sparial_complexity


# 时间复杂度
def cal_time_complexity(swarm, old_swarm):
    time_complexity = 0
    for i in range(len(swarm)):
            # 计算两个智能体之间的速度方向的夹角
        agent_i = swarm[i]
        agent_i_old = old_swarm[i]
        time_complexity = time_complexity + cal_angle_of_vector1([agent_i.vel[0], agent_i.vel[1]],[agent_i_old.vel[0], agent_i_old.vel[1]])
    return time_complexity



# 计算两个方向的夹角
def cal_angle_of_vector(v1, v2):
    # print(v1,v2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    # logging.info("v1_norm_v2_norm:{},{},{}".format(v1,v2,norm(v1) * norm(v2) ))
    cons = dot_product / (norm(v1) * norm(v2))
    if cons < -1:
        cons = -1
    if cons > 1:
        cons = 1
    angle_rad = math.acos(cons)
    return angle_rad

# 直接计算夹角的cos值
def cal_angle_of_vector1(v1, v2):
    # print(v1,v2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    # logging.info("v1_norm_v2_norm:{},{},{}".format(v1,v2,norm(v1) * norm(v2) ))
    cons = dot_product / (norm(v1) * norm(v2))
    return cons




# 随机选择领导者
# 这个只是在集群中选定位置
def get_n_rand(n, p):
    leader_list = set()
    while True:
        leader_list.add(random.randint(0, n - 1))
        if len(leader_list) == math.floor(n * p):
            break
    return leader_list


# 角度旋转
def rotation_matrix_about(v, angle):
    x = v[1] * math.sin(angle) + v[0] * math.cos(angle)
    y = v[1] * math.cos(angle) - v[0] * math.sin(angle)
    return [x, y]


# 定义工具函数，计算两个智能体之间的距离
def cal_distance(agent_a, agent_b):
    distance_vector = agent_a.pos - agent_b.pos
    return norm(distance_vector)


# 定义一个智能体的类
# 包含数据和更新位置
class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        # 位置
        self.pos = np.array([0, 0])
        self.pos[0] = np.random.uniform(70, 90)
        self.pos[1] = np.random.uniform(70, 90)
        # 速度
        self.vel = np.random.uniform(-5, 5, 2)

        # 各个方向的速度分量
        self.vel = self.vel / norm(self.vel) * speed

        # 目标影响权重
        self.w_p = 0.3

        # 目标方向
        self.g = np.array([1, 1]) / norm(np.array([1, 1]))

        # 是否被选为领导
        self.is_leader = False

        # 每个点的吸引邻域集合
        self.neibour_set_attract = []

        # 每个点的排斥邻域集合
        self.neibour_set_repulse = []

        # field_of_view 可修改
        self.field_of_view = 2 * pi 

    def update_position(self, delta_t):
        self.pos = self.pos + self.vel * delta_t
        if self.pos[0] < 0:
            self.pos[0] = self.pos[0] + field.width
        if self.pos[0] > field.width:
            self.pos[0] = self.pos[0] - field.width
        if self.pos[1] > field.height:
            self.pos[1] = self.pos[1] - field.height
        if self.pos[1] < 0:
            self.pos[1] = self.pos[1] + field.height


class Couzin():
    # 初始化
    """
    1. 初始化集群数量
    2. 初始化位置
    3. 初始化速度方向
    4. 初始化速度
    5. 初始化排斥距离
    6. 初始化吸引距离
    7. 初始化角速度
    """

    def __init__(self,N,P, Is_visual = True):
        # 初始化参数
        # 初始化集群中个体数量

        self.n = N
        # 初始化排斥距离
        self.a_minimal_range = 10
        # 初始化吸引距离
        self.attract_range = 100
        # 初始化速度
        self.constant_speed = 7
        # 初始化角速度
        self.theta_dot_max = 5
        # 初始化领导者比例
        self.p = P
        # swarm 生成集群
        self.swarm = []
        print("running")

        [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]
        # agent0 = Agent(0, 7)
        # agent0.pos =np.array([70,70])
        # agent1 = Agent(1, 7)
        # agent1.pos = np.array([80,70])
        # agent2 = Agent(2, 7)
        # agent2.pos = np.array([70,70])

        # agent3 = Agent(3, 7)
        # agent3.pos = np.array([80,80])
        # agent4 = Agent(2, 7)
        # agent4.pos = np.array([75,75])
        # self.swarm = [agent0, agent1, agent2, agent3, agent4]
        # 需要修改，个体位置初始化时候
        # 生成第一个个体的坐标
        self.observation = []
        self.reward = 0

        # 设置时间步长
        self.dt = 0.2

        ##################################################

        # 生成领导者
        # self.leader_list = get_n_rand(self.n, self.p)
        # 固定领导
        self.leader_list = [1,2]
        # print("running1")
        # logging.info("leader_num:{}".format(self.leader_list))

        # 更新领导者标志
        for j in range(len(self.swarm)):
            for leader_id in self.leader_list:
                if j == leader_id:
                    self.swarm[j].is_leader = True
        self.fig = plt.figure()
        self.ax = self.fig.gca()

        #  定义目标点位置
        self.target_x = 450
        self.target_y = 450
        self.target_radius = 50

        # 可视化展示功能开关
        self.is_visual = Is_visual

        # 已经运行的steps数量
        self.total_steps = 0

        #  奖励是总的奖励，而观察室暂态的
        self.reward = 0

        # 存储上一个状态
        self.last_observation = self.swarm

        # 存储空间复杂度
        self.space_complexity = 0

        # 空间-时间复杂度
        self.space_time_flag = True

        # 旧的swarm状态
        self.old_swarm = copy.deepcopy(self.swarm)


        # 存储时间复杂度
        self.time_complexity = 0



    def talker_listener(self,actions):
        
        # 定义callback函数
        def callback(msg):
            rospy.loginfo("respond_node0:%s","\n".join([f"id: {person.id}, x: {person.x}, y: {person.y}, v_x:{person.v_x}, v_y:{person.v_y}" for person in msg.agents]))
            # rospy.loginfo("respond_node1:%s",msg.agents)
            
            # 同步
            # 读取计算出来的速度方向
            for item in msg.agents:
                for j in range(len(self.swarm)):
                    if item.id == self.swarm[j].id:
                        self.swarm[j].vel[0] = item.v_x
                        self.swarm[j].vel[1] = item.v_y

        # 首先定义发布者
        pub = rospy.Publisher('agents1', agents,queue_size=10)
        # 节点定义
        rospy.init_node('node1', anonymous=True)
        # 定义订阅者
        rospy.Subscriber('agents0', agents, callback)
        # 定义发布速率
        rate = rospy.Rate(1)

        self.step(actions)
        
        while not rospy.is_shutdown(): 
            # 定义发布者消息
            agent_list = []
            
            # 测试用 
            for i in range(len(self.swarm)):
                self.swarm[i].vel[0] = self.swarm[i].vel[0] + 2
                self.swarm[i].vel[1] = self.swarm[i].vel[1] + 2

            # 填充agents信息
            # 屏幕上的节点，发布速度方向和位置
            for i in range(len(self.swarm)):
                agent_list.append(agent(self.swarm[i].id,self.swarm[i].pos[0],self.swarm[i].pos[1],
                                    self.swarm[i].vel[0],self.swarm[i].vel[1]))
            pub.publish(agent_list)
            rate.sleep()

    def reset(self):
        # 每次迭代完, 重置swarm, 各个智能体的位置和速度方向
        swarm = []
        [swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]
        # agent0 = Agent(0, 7)
        # agent0.pos =np.array([70,70])
        # agent1 = Agent(1, 7)
        # agent1.pos = np.array([80,70])
        # agent2 = Agent(2, 7)
        # agent2.pos = np.array([70,70])

        # agent3 = Agent(3, 7)
        # agent3.pos = np.array([80,80])
        # agent4 = Agent(2, 7)
        # agent4.pos = np.array([75,75])
        # swarm = [agent0, agent1, agent2, agent3, agent4]

        self.swarm = swarm

        # self.leader_list = get_n_rand(self.n, self.p)
        # 还用之前的领导者个体
        # 更新领导者标志
        for j in range(len(self.swarm)):
            for leader_id in self.leader_list:
                if j == leader_id:
                    self.swarm[j].is_leader = True
        obs_ = [[] for _ in range(self.n)]
       # 添加功能,返回各个点的观察域
        for i in range(len(self.swarm)):
            agent = self.swarm[i]
            for j in range(len(self.swarm)):
                neighbor = self.swarm[j]
                visual_vector = np.array([neighbor.pos[0] - agent.pos[0], neighbor.pos[1] - agent.pos[1]])
                if agent.id != neighbor.id and cal_distance(agent,
                                                            neighbor) < self.attract_range and cal_angle_of_vector(
                    visual_vector, agent.vel) < agent.field_of_view / 2:

                    # 位置向量，单位位置向量，距离
                    r = neighbor.pos - agent.pos
                    if norm(r) == 0:
                        r_normalized = r
                    else:
                        r_normalized = r / norm(r)
                    # 位置向量标准化
                    norm_r = norm(r)
                    # 速度向量              
                    if norm_r < self.a_minimal_range:
                        # 添加排斥区域
                        agent.neibour_set_repulse.append(neighbor)
                    elif norm_r < self.attract_range:
                        # 添加吸引区域邻域集合
                        # logging.info("{}:adding".format(agent.id))
                        agent.neibour_set_attract.append(neighbor)
            obs_single = [[] for _ in range(self.n)]
            obs_single[0] = [0, 0, agent.vel[0], agent.vel[1]]
            p = 1
            for item in agent.neibour_set_attract:
                obs_single[p] = [item.pos[0] - agent.pos[0], item.pos[1] - agent.pos[1], item.vel[0], item.vel[1]]
                p = p + 1
            # 多余补0
            for m in range(len(obs_single)):
                if len(obs_single[m]) == 0:
                    obs_single[m] = [0, 0, 0, 0]
            obs_[i] = obs_single
        # obs_ 的转化
        # logging.info("obs_:{}".format(obs_))
        obs = convert_list2(obs_)
        # logging.info("obs:{}".format(obs))
        # 在reset时候，同时需要将reward重置
        self.reward = 0
        # 运行次数也得重置
        self.total_steps = 0

        # 重置空间-时间标志
        self.space_time_flag = True
        
        self.space_complexity = 0 

        self.time_complexity = 0

        return obs

    # 核心函数
    # 奖励函数-运动趋势
    # 分裂-惩罚 平均空间相关度
    # 整体reward 到达目标点大的reward
    # 每个step要输入action，只有输入action后才能输出序列，每个个体要输出一个可视角
    # 增加actions, 给每个个体增加可视角集合 actions = [a1, a2, a3, a4]

    def step(self, actions):    
        # 更新个体的位置
        # 不用进行处理更新，直接进行更新就好
        for v in range(len(self.swarm)):
            agent.update_position(self.dt)
        
        
        
        
        # # actions 是一个集合，包含追随者的可视角和领综合
        # # actions是个混合动作集合，包括可视角和影响权重
        # # obs_ 存储每个个体观察区的个体位置(pos,vel)，
        # obs_ = [[] for _ in range(self.n)]
        # # 到达目标的个体的数量
        # count_fn = 0
        # # 遍历集群
        # for i in range(len(self.swarm)):
        #     agent = self.swarm[i]
        #     # 清空排斥区域/吸引区域
        #     agent.neibour_set_attract = []
        #     agent.neibour_set_repulse = []
        #     # 2005 couzin领导模型
        #     d = 0
        #     # 排斥域
        #     dr = 0
        #     # 吸引域
        #     da = 0
        #     # 当前个体的速度
        #     dv = agent.vel

        #     # 更新各个个体的可视角
        #     # 先判断是否是领导者，是领导者的话静态可视角，动态影响权重
        #     # 非领导者的话，无影响权重动态可视角
        #     if i in self.leader_list:
        #         agent.field_of_view = 2 * math.pi
        #         agent.w_p = actions[i]
        #         # logging.info("w_p:{}".format(agent.w_p))
        #     else:
        #         agent.field_of_view = actions[i]

        #     if agent.is_leader:
        #         agent.g = np.array([self.target_x, self.target_y]) - agent.pos
        #         agent.g = agent.g / norm(agent.g)

        #     neighbor_count = 0

        #     # 这边要做个判断，如果已经到达目标范围内，则直接更新速度方向
        #     if math.sqrt(
        #             pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) < self.target_radius:
        #         agent.vel = np.array([self.target_x - agent.pos[0], self.target_y - agent.pos[1]])
        #         agent.vel = agent.vel / norm(agent.vel) * self.constant_speed
        #         count_fn = count_fn  +  1
        #     else:
        #         for j in range(len(self.swarm)):
        #             neighbor = self.swarm[j]
        #             visual_vector = np.array([neighbor.pos[0] - agent.pos[0], neighbor.pos[1] - agent.pos[1]])
        #             # 可视性检查
        #             # logging.info(":{},{},{}".format(agent.id != neighbor.id,cal_distance(agent,
        #             #                                             neighbor) < self.attract_range and cal_angle_of_vector(
        #             #     visual_vector, agent.vel) < agent.field_of_view / 2, math.sqrt(
        #             # pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) > self.target_radius))
                    
        #             # 计算速度方向夹角 
        #             th1 = cal_angle_of_vector1(agent.vel, neighbor.vel)
        #             angle = (-1/4) * math.pi * th1 + math.pi
        #             # logging.info("angle:{}".format(angle))
        #             if agent.id != neighbor.id and cal_distance(agent,
        #                                                         neighbor) < self.attract_range and cal_angle_of_vector(
        #                 visual_vector, agent.vel) < angle and math.sqrt(
        #             pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) > self.target_radius:

        #                 neighbor_count = neighbor_count + 1
        #                 # 位置向量，单位位置向量，距离
        #                 r = neighbor.pos - agent.pos
        #                 # logging.info("r:{}".format(r))
        #                 r_normalized = 0
        #                 if norm(r) == 0:
        #                     r_normalized = r
        #                 else:
        #                     r_normalized = r / norm(r)
        #                 # 位置向量标准化
        #                 norm_r = norm(r)

        #                 # 通过actions 给每个个体可视角赋值

        #                 # 速度向量
        #                 agent_vel_normalized = agent.vel / norm(agent.vel)
        #                 if cal_angle_of_vector(r_normalized, agent_vel_normalized) < agent.field_of_view / 2:
        #                     if norm_r < self.a_minimal_range:
        #                         # logging.info("{}:rejecting".format(agent.id))
        #                         # 添加排斥区域
        #                         agent.neibour_set_repulse.append(neighbor)
        #                         # 排斥区域，位置累计
        #                         dr = dr - r_normalized
        #                     elif norm_r < self.attract_range:
        #                         # 添加吸引区域邻域集合
        #                         # logging.info("{}:adding11".format(agent.id))
        #                         agent.neibour_set_attract.append(neighbor)
        #                         # 吸引区域位置向量累计
        #                         da = da + r_normalized
        #                         # 吸引区速度向量累计
        #                         dv = dv + neighbor.vel / norm(neighbor.vel)
        #         if norm(dr) != 0:
        #             # 排斥区域
        #             # if agent.is_leader:
        #             #     dr = dr / norm(dr)
        #             #     d = (dr + agent.w_p * agent.g) / norm(dr + agent.w_p * agent.g)
        #             # else:
        #             #     d = dr / norm(dr)
        #             d = dr / norm(dr)
        #         elif norm(da) != 0:
        #             # 吸引区域
        #             if agent.is_leader:
        #                 # 计算周边个体的平均位置
        #                 neiX = 0
        #                 neiY = 0
        #                 K = len(agent.neibour_set_attract)
        #                 for i in range(len(agent.neibour_set_attract)):
        #                     neiX = agent.neibour_set_attract[i].pos[0]
        #                     neiY = agent.neibour_set_attract[i].pos[1]
        #                 nei_d = math.sqrt(math.pow((agent.pos[0] - neiX / K), 2) + math.pow((agent.pos[1] - neiY / K), 2))
        #                 agent.w_p =  math.exp((couzin.n / 25) * (-nei_d / couzin.attract_range))
        #                 # logging.info("w_p:{}".format(agent.w_p))
               
        #                 d_new = (da + dv) / norm(da + dv)
        #                 d = (d_new + agent.w_p * agent.g) / norm(d_new + agent.w_p * agent.g)
        #             else:
        #                 d_new = (da + dv) / norm(da + dv)
        #                 d = d_new
        #         else:
        #             if i in self.leader_list:
        #                 agent.vel = agent.g * self.constant_speed

        #         if norm(d) != 0:
        #             angle_between = cal_angle_of_vector(d, agent.vel)
        #             # logging.info("angle_between:{}".format(angle_between))
        #             if angle_between >= self.theta_dot_max * self.dt:
        #                 # rotation_matrix_about 旋转后，返回的是向量
        #                 rot = rotation_matrix_about(agent.vel, self.theta_dot_max * self.dt)

        #                 vel0 = rot

        #                 rot1 = rotation_matrix_about(agent.vel, -self.theta_dot_max * self.dt)

        #                 vel1 = rot1

        #                 if cal_angle_of_vector(vel0, d) < cal_angle_of_vector(vel1, d):
        #                     agent.vel = vel0 / norm(vel0) * self.constant_speed
        #                 else:
        #                     agent.vel = vel1 / norm(vel1) * self.constant_speed
        #             else:
        #                 agent.vel = d * self.constant_speed

        #     # 将邻居信息更新在obs_single中
        #     # 将单个个体的观察空间长度固定
        #     # 修改的地方在于加上本智能体的信息，在考虑与周边个体的关系时，同时需要本个体的位置和速度信息
        #     obs_single = [[] for _ in range(self.n)]
        #     # logging.info(
        #     #     "attract:{}".format(agent.neibour_set_attract))
        #     obs_single[0] = [0, 0, agent.vel[0], agent.vel[1]]          
        #     p = 1
        #     for item in agent.neibour_set_attract:
        #         # logging.info("p:{},{},{},{},{}".format(agent.id, p, len(obs_single), len(agent.neibour_set_attract), item.id))

        #         obs_single[p] = [item.pos[0] - agent.pos[0], item.pos[1] - agent.pos[1], item.vel[0], item.vel[1]]
        #         p = p + 1
        #     # 多余补0
        #     for m in range(len(obs_single)):
        #         if len(obs_single[m]) == 0:
        #             obs_single[m] = [0, 0, 0, 0]

        #     obs_[i]  = obs_single
        #     # logging.info("obs_[i]:{}".format(len(obs_[i])))
        # # 更新各个点的坐标位置
        # [agent.update_position(self.dt) for agent in self.swarm]
        # # 输出各个智能体的编号，坐标，速度方向,是否是领导者
        # # logging.info("#########################")
        # # for i in range(len(self.swarm)):
        # #     logging.info("swarm:{},{},{}".format(self.swarm[i].id, self.swarm[i].pos, self.swarm[i].vel))

        # if self.is_visual:
        #     # 可视化展示

        #     x = np.array([])
        #     y = np.array([])

        #     x_dot = np.array([])
        #     y_dot = np.array([])

        #     for agent in self.swarm:
        #         # 存储所有横纵坐标位置
        #         # x，y分别存储x,y
        #         x = np.append(x, agent.pos[0])
        #         y = np.append(y, agent.pos[1])

        #         # 存储所有横纵速度方向
        #         # x_dot，y_dot 分别存储x,y
        #         x_dot = np.append(x_dot, agent.vel[0])
        #         y_dot = np.append(y_dot, agent.vel[1])

        #     # 清除展示区
        #     self.ax.clear()
        #     # logging.info("x:{}".format(x))
        #     # logging.info("y:{}".format(y))
        #     # logging.info("x_dot:{}".format(x_dot))
        #     # logging.info("y_dot:{}".format(y_dot))
        #     # 设置箭头的形状和大小
        #     # 追随者
        #     x_temp = np.array([])
        #     y_temp = np.array([])
        #     x_temp_dot = np.array([])
        #     y_temp_dot = np.array([])

        #     # 领导者
        #     x_temp_f = np.array([])
        #     y_temp_f = np.array([])
        #     x_temp_dot_f = np.array([])
        #     y_temp_dot_f = np.array([])

        #     # logging.info("self.leader_list:{}".format(self.leader_list))
        #     for i in range(len(self.swarm)):
        #         if i not in list(self.leader_list):
        #             # x，y分别存储x,y方向上的位置
        #             x_temp = np.append(x_temp, self.swarm[i].pos[0])
        #             y_temp = np.append(y_temp, self.swarm[i].pos[1])

        #             # x_dot，y_dot 分别存储x,y方向上的范数
        #             x_temp_dot = np.append(x_temp_dot, self.swarm[i].vel[0] / norm(self.swarm[i].vel) * 0.4)
        #             y_temp_dot = np.append(y_temp_dot, self.swarm[i].vel[1] / norm(self.swarm[i].vel) * 0.4)

        #     self.ax.quiver(x_temp, y_temp, x_temp_dot, y_temp_dot, width=0.01,
        #                    scale=5, units="inches", color='#EC3684', angles='xy')

        #     for item in self.leader_list:
        #         # x，y分别存储x,y方向上的位置

        #         x_temp_f = np.append(x_temp_f, self.swarm[item].pos[0])
        #         # logging.info("x_temp_f:{}".format(x_temp_f))
        #         y_temp_f = np.append(y_temp_f, self.swarm[item].pos[1])

        #         # x_dot，y_dot 分别存储x,y方向上的范数
        #         x_temp_dot_f = np.append(x_temp_dot_f, self.swarm[item].vel[0] / norm(self.swarm[item].vel) * 0.4)
        #         y_temp_dot_f = np.append(y_temp_dot_f, self.swarm[item].vel[1] / norm(self.swarm[item].vel) * 0.4)
        #     # logging.info("x_temp_f:{}".format(x_temp_f))

        #     self.ax.quiver(x_temp_f, y_temp_f, x_temp_dot_f,
        #                    y_temp_dot_f,
        #                    width=0.01, scale=5, units="inches", color='#006400', angles='xy')

        #     # 添加画线, 画出排斥和吸引，判断是否正常运行
        #     for k in range(len(self.swarm)):
        #         # 画self.swarm[k] 与其邻居的线
        #         # 创建邻居集合
        #         neibors_attrack = []
        #         for m in range(len(self.swarm[k].neibour_set_attract)):
        #             neibors_attrack.append(
        #                 [self.swarm[k].neibour_set_attract[m].pos[0], self.swarm[k].neibour_set_attract[m].pos[1]])
        #         if len(neibors_attrack) != 0:
        #             x_points, y_points = zip(*neibors_attrack)
        #             # 绘制连线
        #             for x, y in neibors_attrack:
        #                 plt.plot([self.swarm[k].pos[0], x], [self.swarm[k].pos[1], y], 'g--', linewidth=0.1, zorder=1)
        #         # 排斥区域邻居集合
        #         neibors_repluse = []
        #         for m in range(len(self.swarm[k].neibour_set_repulse)):
        #             neibors_repluse.append(
        #                 [self.swarm[k].neibour_set_repulse[m].pos[0], self.swarm[k].neibour_set_repulse[m].pos[1]])
        #         if len(neibors_repluse) != 0:
        #             x_points, y_points = zip(*neibors_repluse)
        #             # 绘制连线
        #             for x, y in neibors_repluse:
        #                 plt.plot([self.swarm[k].pos[0], x], [self.swarm[k].pos[1], y], 'r--', linewidth=0.1, zorder=1)

        #     self.ax.set_aspect('auto', 'box')
        #     self.ax.set_xlim(0, field.width)
        #     self.ax.set_ylim(0, field.height)

        #     self.ax.tick_params(axis='x', colors='red')
        #     self.ax.tick_params(axis='y', colors='blue')
        #     circle = plt.Circle((self.target_x, self.target_y), self.target_radius, color='r', fill=False)
        #     plt.gcf().gca().add_artist(circle)
        #     plt.pause(0.01)
        # # 清除个体的邻居集合
        # for n in range(len(self.swarm)):
        #     self.swarm[n].neibour_set_attract = []
        #     self.swarm[n].neibour_set_repulse = []

        # connect_value = self.connectivity_cal()
        # # logging.info("connect_value:{}".format(connect_value))

        # self.total_steps = self.total_steps + 1
        # # 计算时间复杂度
        # if self.total_steps > 1:
        #     self.time_complexity  = cal_time_complexity(self.swarm, self.old_swarm) + self.time_complexity
        # # 更新old_swarm
        # self.old_swarm = copy.deepcopy(self.swarm)

        # self.space_complexity = cal_space_complexity(self.swarm) + self.space_complexity
      
        # # 奖励函数设计, observation的设计
        # # 连通度设计奖励，到达奖励的设计
        # # 连通度奖励就以连通度为奖励
        # # 到达终点时的奖励设计
        # """
        #    在某个时刻到达终点的个数越多奖励越大 
        # """
        # arrival_rate = self.arrival_proportion_cal()
        # # with open("connect_value.txt","a+") as space:
        # #     space.write(str(connect_value)+"\n")        

        # # reward 为每一步取得的奖励
        # self.reward = connect_value + arrival_rate * 50
        # done = [False] * self.n
        # # 3个结束条件: 1.总的仿真次数  2.分裂，如果已经提前分裂，则无必要继续训练 3. 到达率到达0.7以上
        # evaluation = True
        # if not evaluation:
        #     if self.total_steps > 2000 or arrival_rate > 0.7 or connect_value < 0.2:
        #         # print("total_step:",self.total_steps, " connect_value:", connect_value, " arrival_rate:", arrival_rate)
        #         done = [True] * self.n
        # else:
        #     if self.total_steps >= 2000:
        #     # print("total_step:",self.total_steps, " connect_value:", connect_value, " arrival_rate:", arrival_rate)
        #         done = [True] * self.n
        # # logging.info("obs_before:{}".format(obs_))
        # obs1_ = convert_list2(obs_)
        # # logging.info("obs_after:{}".format(obs1_))
        # # 给每个个体不同的奖励，周边的个体少的给少的奖励
        # # 回报函数设计
        
        # reward_temp = []
        # # 奖励函数1
        # # 周边个体的数量奖励和到达目标的奖励
        # for i in range(len(obs_)):
        #     num = 0
        #     for item1 in obs_[i]:              
        #         for item2 in item1:
        #             if item2 !=0:
        #                 num = num + 1
        #                 break
        #     distance = math.sqrt(pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2))
        #     if distance < self.target_radius:
        #         current_reward = 20
        #         if num - 1 == 0:
        #             reward_temp.append(current_reward * 0.5)
        #         else:
        #             reward_temp.append(current_reward* (num - 1))  
        #     else:
        #         if num - 1 == 0:
        #             reward_temp.append(-1)
        #         else:
        #             reward_temp.append(num - 1)

  
        # # 奖励函数2 - 与集群中心距离越近，奖励越大，越远奖励越小
        # center_x = 0
        # center_y = 0
        # for i in range(len(self.swarm)):
        #     center_x = self.swarm[i].pos[0] + center_x
        #     center_y = self.swarm[i].pos[1] + center_y
        # center_x = center_x / len(self.swarm)
        # center_y = center_y / len(self.swarm)
        # if(all(done)==True and self.space_time_flag == True) or (self.space_time_flag == True and count_fn > 0):
        #     space_complexity = self.space_complexity / self.total_steps / (self.n * (self.n - 1) /2)
        #     time_comlexity = self.time_complexity / (self.total_steps - 1) / self.n

        #     # 写入空间复杂度和时间复杂度
        #     # with open("space_complexity.txt","a+") as space:
        #     #     space.write(str(space_complexity)+"\n") 
        #     # with open("time_complexity.txt","a+") as space:
        #     #     space.write(str(time_comlexity)+"\n") 

        #     self.space_time_flag = False 
        #     # print("time_comlexity",time_comlexity)  
        # # logging.info("resward_env:{}".format(reward))

        # return obs1_,  reward_temp, done,


    def connectivity_cal(self):
        connectivity = 0

        # 计算连通度
        class Uav:
            def __init__(self, id, p_x, p_y):
                self.id = id
                self.p_x = p_x
                self.p_y = p_y

        def is_in(l1, l2):
            for item in l2:
                if l1.id == item.id:
                    return True
            return False

        def is_all_adjacent_nodes_visited(temp, layer, ls):
            for item in layer:
                # logging.info("ls:{}".format(print_f(ls)))
                adjacent_nodes = find_adjacent_nodes(item, ls)
                # logging.info(" adjacent_nodes40:{}".format(print_f(adjacent_nodes)))
                for item1 in adjacent_nodes:
                    cons = False
                    for item2 in temp:
                        if item1.id == item2.id:
                            cons = True
                    if not cons:
                        return False
            return True

        def find_adjacent_nodes(node, agent_set):
            adjacent_nodes = []
            for agent in agent_set:
                if node.id != agent.id:
                    if is_connected(node, agent):
                        adjacent_nodes.append(agent)
            return adjacent_nodes

        def is_connected(a1, a2):
            if math.sqrt((a1.p_x - a2.p_x) ** 2 + (a1.p_y - a2.p_y) ** 2) <= self.attract_range:
                return True
            return False

        def cal_cluster(ls):
            clusters = []
            while len(ls) != 0:
                temp = ls[0]
                # logging.info("temp:{}".format(temp.id))
                temp1 = [temp]
                adjacent_nodes = [temp]
                while not is_all_adjacent_nodes_visited(temp1, adjacent_nodes, ls):
                    adjacent_nodes_temp = []
                    for node in adjacent_nodes:
                        adjacent_nodes1 = find_adjacent_nodes(node, ls)
                        # logging.info("adjacent_nodes1:{}".format(print_f(adjacent_nodes1)))
                        for adjacent_node in adjacent_nodes1:
                            if is_in(adjacent_node, temp1):
                                continue
                            # temp1是一个子群的点
                            temp1.append(adjacent_node)
                            # 邻居节点添加到所在层
                            adjacent_nodes_temp.append(adjacent_node)
                    # adjacent_nodes_temp 新的层的邻居的集合
                    adjacent_nodes = adjacent_nodes_temp
                # logging.info("temp1:{}".format(print_f(temp1)))
                clusters.append(temp1)

                i = 0
                while i < len(ls):
                    j = 0
                    while j < len(temp1):
                        if ls[i].id == temp1[j].id:
                            ls.pop(i)
                            i -= 1  # 减少 i 以便继续检查当前位置
                            break
                        j += 1
                    i += 1

                # item_set = []
                # for item in ls:
                #     item_set.append([item.id,item.p_x,item.p_y])
                # logging.info("ls:{}".format(item_set))

            return clusters

        # 连通度计算
        # 通过deepcopy 创建UAV集合
        uavs = []
        for item in self.swarm:
            uavs.append(Uav(item.id, item.pos[0], item.pos[1]))
        # 计算分群
        cons_flock = cal_cluster(uavs)
        # 创建领导者id群
        leader_id_set = []
        for item in self.leader_list:
            leader_id_set.append(self.swarm[item].id)
        sub_swarm_set = []
        for sub_flock in cons_flock:
            for sub_flock_item in sub_flock:
                if sub_flock_item.id in leader_id_set:
                    break
            sub_swarm_set.append(sub_flock)

        for flock_contains_leadr in sub_swarm_set:
            subswarm_length = len(flock_contains_leadr)
            if subswarm_length != 0:
                connectivity = connectivity + subswarm_length * (subswarm_length - 1)
        connectivity = connectivity / (len(self.swarm) * (len(self.swarm) - 1))
        return connectivity

    def arrival_proportion_cal(self):
        arrival_num = 0
        for point in self.swarm:
            if math.sqrt(
                    (point.pos[0] - self.target_x) ** 2 + (point.pos[1] - self.target_y) ** 2) < self.attract_range:
                arrival_num = arrival_num + 1
        return arrival_num / len(self.swarm)






def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"File '{file_path}' does not exist.")





if __name__ == '__main__':

    # import os
    
    # file_path_space = "space_complexity.txt"
    # file_path_time = "time_complexity.txt"
    # file_path_connect = "connect_value.txt"
    # try:
    #     delete_file_if_exists(file_path_space)
    #     delete_file_if_exists(file_path_time)
    #     delete_file_if_exists(file_path_connect)
    # except OSError as e:
    #     print(f"删除文件时发生错误: {e}")

    couzin = Couzin(5,0.2,False)
    couzin.attract_range = 60

    leaders = couzin.leader_list
    w = 0.5
    angle = 2 * math.pi
    actions = []
    logging.info("runnning")
    for i in range(len(couzin.swarm))     :
        if i in leaders:
            actions.append(w)
        else:
            actions.append(angle)

    couzin.talker_listener(actions)
