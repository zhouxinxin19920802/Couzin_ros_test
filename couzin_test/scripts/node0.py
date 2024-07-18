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
# 韧性平滑用
from scipy.signal import savgol_filter as sg
import copy 
from resilience import get_min,calculate_fluctuation
# import rospy
import copy
import os

# linux
# import ros

# 信息列表
# from couzin_test.msg import agent
# from couzin_test.msg import agents

# from maddpg_copy0 import MADDPG
# from turtle_control.msg import vs_cmd, v_cmd
# from tuio.msg import obj




# import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *

logging.basicConfig(
    level=logging.INFO,  # 控制台打印的日志级别
    filename="test_log.txt",
    filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Field:
    def __init__(self):
        self.width = 2000
        self.height = 1000


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
    sparial_complexity =sparial_complexity

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
        self.pos[0] = np.random.uniform(40, 60)
        self.pos[1] = np.random.uniform(40, 60)
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
        
        
        # 小车和桌面的夹角
        self.angle_car_table = 0
        
    # TODO 修改不穿墙而过
    # def update_position(self, delta_t):
    #     self.pos = self.pos + self.vel * delta_t
    #     if self.pos[0] < 0:
    #         self.pos[0] = self.pos[0] + field.width
    #     if self.pos[0] > field.width:
    #         self.pos[0] = self.pos[0] - field.width
    #     if self.pos[1] > field.height:
    #         self.pos[1] = self.pos[1] - field.height
    #     if self.pos[1] < 0:
    #         self.pos[1] = self.pos[1] + field.height

    # 触碰墙的时候速度方向相反
    def update_position(self, delta_t):
        
        previous_loation = self.pos
        
        self.pos = self.pos + self.vel * delta_t
        if self.pos[0] < 0 or self.pos[0] > field.width:
            self.vel[0] = - self.vel[0]
            self.pos = self.pos + self.vel * delta_t
        if self.pos[1] > field.height or self.pos[1] < 0:
            self.vel[1] =  - self.vel[1]
            self.pos = self.pos + self.vel * delta_t
            
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

    def __init__(self,N, P, Is_visual = True):
        # 初始化参数
        # 初始化集群中个体数量

        self.n = N
        # 初始化排斥距离
        self.a_minimal_range = 10
        # 初始化吸引距离
        self.attract_range = 60
        # 初始化速度
        self.constant_speed = 1
        # 初始化角速度
        self.theta_dot_max = 5
        # 初始化领导者比例
        self.p = P
        # swarm 生成集群
        self.swarm = []         
        def generate_initial_state(self):
            agent0 = Agent(0, self.constant_speed)
            self.swarm.append(agent0)
            for i in range(1,self.n):          
                connection_flag = True
                while(connection_flag):
                    agenti =  Agent(i, self.constant_speed)
                    location_flag = True
                    for item in self.swarm:
                        if (agenti.pos[0] == item.pos[0] and agenti.pos[1] == item.pos[1]):
                            location_flag = False
                            break
                
                    if location_flag == True:
                        if math.sqrt((agenti.pos[0] - item.pos[0]) * (agenti.pos[0] - item.pos[0]) + 
                                    (agenti.pos[0] - item.pos[1]) * (agenti.pos[0] - item.pos[1])) < self.attract_range:
                            connection_flag = False
                            self.swarm.append(agenti)
            return self.swarm
        # [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]
        self.swarm = generate_initial_state(self)

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

        # 写入一次
        self.write_once_flag = True

        # 最大子群
        self.max_sub_swarm = []
        
        
    def talker_listener(self, actions):
        ##############################
        # 强化学习

        ##############################

        # 节点定义
        rospy.init_node("node0", anonymous=True)
        
        # 定义callback函数
        def callback(msg):
            """rospy.loginfo(
                "respond_node1:%s",
                "\n".join(
                    [
                        f"id: {person.id}, x: {person.x}, y: {person.y}, v_x:{person.v_x}, v_y:{person.v_y}"
                        for person in msg.agents
                    ]
                ),
            )"""
            
            global total_messaage, flag_once_, flag_
            # 同步
           
            total_messaage[msg.ID] = [msg.x, msg.y, msg.angle]
            #print(total_messaage)
            if len(set(total_messaage.keys())) == self.n:
                if flag_once_:
                    for j in range(len(self.swarm)):
                        # for item in total_messaage:
                            # print("item:{}".format(item))
                        self.swarm[j].id = list(total_messaage.keys())[j]
                    flag_once_ = False
                #print(self.swarm[0].id,self.swarm[1].id,self.swarm[2].id)
                for j in range(len(self.swarm)):
                    for item in total_messaage:
                        if item == self.swarm[j].id:
                            self.swarm[j].pos[0] = total_messaage[item][0]
                            self.swarm[j].pos[1] = total_messaage[item][1]
                            
                            # TODO 读取消息中小车的速度方向和桌面的夹角，需要确认下是第几位
                            self.swarm[j].angle_car_table = total_messaage[item][3]
                            
                            
                            self.swarm[j].vel[0] = (
                                math.cos(total_messaage[item][2]) * self.constant_speed
                            )
                            self.swarm[j].vel[1] = (
                                math.sin(total_messaage[item][2]) * self.constant_speed
                            )
                #print(self.swarm[0].id,self.swarm[1].id,self.swarm[2].id)
                self.step(actions)
                total_messaage = {}
                rospy.loginfo_once("接收到屏幕小车信息,成功进入callback循环")
                flag_ = True

            """ 
            强化学习                                
            actions = maddpg_agents_.choose_action(obs)
            obs1_,  reward_temp, done = couzin.step(actions=actions)
            obs = obs1_
            """

        global flag_
        pub_set = []
        # 首先定义发布者
        for i in range(11):
            pub = rospy.Publisher(
                "robot_" + str(i) + "_control/v_cmd", v_cmd, queue_size=10
            )
            pub_set.append(pub)

        # 定义订阅者
        rospy.Subscriber("tuio_obj", obj, callback)
        # 定义发布速率
        rate = rospy.Rate(1)
        
        while not rospy.is_shutdown():
            # 定义发布者消息
            # # 测试用
            # for i in range(len(self.swarm)):
            #     self.swarm[i].vel[0] = self.swarm[i].vel[0] + 1
            #     self.swarm[i].vel[1] = self.swarm[i].vel[1] + 1

            # 填充agents信息
            # 计算节点发布速度方向
            if flag_:
                for i in range(len(self.swarm)):
                    msg_ = v_cmd()
                    # msg_.v_x = self.swarm[i].vel[0]
                    # msg_.v_y = self.swarm[i].vel[1]
                    # 世界坐标系到小车坐标系的转换
                    msg_.v_x = self.swarm[i].vel[0] * math.cos(self.swarm[i].angle_car_table) + self.swarm[i].vel[1] * math.sin(self.swarm[i].angle_car_table)
                    msg_.v_y = -self.swarm[i].vel[0] * math.sin(self.swarm[i].angle_car_table) + self.swarm[i].vel[1] * math.cos(self.swarm[i].angle_car_table)             
                    #print(self.swarm[i].id)
                    pub_set[self.swarm[i].id].publish(msg_)
                    rospy.loginfo("id:%d, x速度:%f,y速度:%f",self.swarm[i].id,self.swarm[i].vel[0],self.swarm[i].vel[1])
                flag_ = False

                # agent_list.append(
                #     agent(
                #         self.swarm[i].id,
                #         0,
                #         0,
                #         self.swarm[i].vel[0],
                #         self.swarm[i].vel[1],
                #     )
                # )
                # vs_cmd.v_x[self.swarm[i].id] = self.swarm[i].vel[0]
                # vs_cmd.v_x[self.swarm[i].id] = self.swarm[i].vel[1]
            rate.sleep()

        
    def reset(self):
        # 每次迭代完, 重置swarm, 各个智能体的位置和速度方向
        self.swarm = []
        def generate_initial_state(self):
            agent0 = Agent(0, self.constant_speed)
            self.swarm.append(agent0)
            for i in range(1,self.n):          
                connection_flag = True
                while(connection_flag):
                    agenti =  Agent(i, self.constant_speed)
                    location_flag = True
                    for item in self.swarm:
                        if (agenti.pos[0] == item.pos[0] and agenti.pos[1] == item.pos[1]):
                            location_flag = False
                            break
                
                    if location_flag == True:
                        if math.sqrt((agenti.pos[0] - item.pos[0]) * (agenti.pos[0] - item.pos[0]) + 
                                    (agenti.pos[0] - item.pos[1]) * (agenti.pos[0] - item.pos[1])) < self.attract_range:
                            connection_flag = False
                            self.swarm.append(agenti)
            return self.swarm
        # [self.swarm.append(Agent(i, self.constant_speed)) for i in range(self.n)]
        self.swarm = generate_initial_state(self)
        # cons1 = []
        # for item in self.swarm:
        #     cons1.append([item.id,item.pos])
        # logging.info("cons1:{}".format(cons1))

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

        self.write_once_flag = True

        return obs

    # 核心函数
    # 奖励函数-运动趋势
    # 分裂-惩罚 平均空间相关度
    # 整体reward 到达目标点大的reward
    # 每个step要输入action，只有输入action后才能输出序列，每个个体要输出一个可视角
    # 增加actions, 给每个个体增加可视角集合 actions = [a1, a2, a3, a4]

    def step(self, actions):       
        # 到达目标的个体的数量
        count_fn = 0
        count_fn_leader = 0
        obs_ = [[] for _ in range(self.n)]
        # 遍历集群
        if self.total_steps <= 200:
            failure_list = []
        else:
            failure_list = []
        for i in range(len(self.swarm)):
            agent = self.swarm[i]
            # 不在异常个体列表
            if agent.id not in failure_list:
            
            # 清空排斥区域/吸引区域
                agent.neibour_set_attract = []
                agent.neibour_set_repulse = []
                # 2005 couzin领导模型
                d = 0
                # 排斥域
                dr = 0
                # 吸引域
                da = 0
                # 当前个体的速度
                dv = agent.vel

                # 更新各个个体的可视角
                # 先判断是否是领导者，是领导者的话静态可视角，动态影响权重
                # 非领导者的话，无影响权重动态可视角
                if i in self.leader_list:
                    agent.w_p = actions[i]
                    # logging.info("w_p:{}".format(agent.w_p))
                else:
                    # andalusia 许之前的方法，追随者是可视角
                    agent.field_of_view = actions[i]

                if agent.is_leader:
                    agent.g = np.array([self.target_x, self.target_y]) - agent.pos
                    agent.g = agent.g / norm(agent.g)
                
                neighbor_count = 0
                
                if math.sqrt(
                        pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) < self.target_radius:
                    agent.vel = np.array([self.target_x - agent.pos[0], self.target_y - agent.pos[1]])
                    agent.vel = agent.vel / norm(agent.vel) * self.constant_speed
                    count_fn = count_fn  +  1
                    if agent.is_leader:
                        count_fn_leader = count_fn_leader  +  1
                else:
                    
                    for j in range(len(self.swarm)):              
                        neighbor = self.swarm[j]
                        visual_vector = np.array([neighbor.pos[0] - agent.pos[0], neighbor.pos[1] - agent.pos[1]])
                        
                        
                        th1 = cal_angle_of_vector1(agent.vel, neighbor.vel)
                        angle = (-1/4) * math.pi * th1 + math.pi
                        
                        # # 原始的couzin模型
                        # angle = agent.field_of_view
                                        
                        # # 计算速度方向夹角 
                        # angle = 2 * math.pi
                        if agent.id != neighbor.id and cal_distance(agent,neighbor) < self.attract_range and cal_angle_of_vector(
                                                visual_vector, agent.vel) < angle and math.sqrt(
                                            pow(agent.pos[0] - self.target_x, 2) + pow(agent.pos[1] - self.target_y, 2)) > self.target_radius:
                            # 位置向量，单位位置向量，距离
                            r = neighbor.pos - agent.pos
                            r_normalized = 0
                            if norm(r) == 0:
                                r_normalized = r
                            else:
                                r_normalized = r / norm(r)
                            # 位置向量标准化
                            norm_r = norm(r)

                            # 通过actions 给每个个体可视角赋值
                            # 速度向量
                            agent_vel_normalized = agent.vel / norm(agent.vel)
                            if cal_angle_of_vector(r_normalized, agent_vel_normalized) < agent.field_of_view / 2:
                                if norm_r < self.a_minimal_range:
                                    # 添加排斥区域
                                    agent.neibour_set_repulse.append(neighbor)
                                    # 排斥区域，位置累计
                                    dr = dr - r_normalized
                                elif norm_r < self.attract_range:
                                    # 在这里进行权重分配计算
                                    # 添加吸引区域邻域集合
                                    # logging.info("agent.attention:{},{}".format(agent.attention, j))
                                    agent.neibour_set_attract.append(neighbor)
                                    # logging.info("len(agent.neibour_set_attract):{}".format(len(agent.neibour_set_attract)))
                                    # 吸引区域位置向量累计
                                    da = da + r_normalized
                                    # 吸引区速度向量累计
                                    dv = dv + neighbor.vel / norm(neighbor.vel)
                    if norm(dr) != 0:
                        d = dr / norm(dr)
                    elif norm(da) != 0:
                        # 吸引区域
                        if agent.is_leader:
                            
                            neiX = 0
                            neiY = 0
                            K = len(agent.neibour_set_attract)
                            for k in range(len(agent.neibour_set_attract)):
                                neiX = agent.neibour_set_attract[k].pos[0]
                                neiY = agent.neibour_set_attract[k].pos[1]
                            nei_d = math.sqrt(math.pow((agent.pos[0] - neiX / K), 2) + math.pow((agent.pos[1] - neiY / K), 2))
                            agent.w_p =  math.exp((couzin.n / 25) * (-nei_d / couzin.attract_range))
                            
                            # 使用一个固定的领导者影响权重
                            agent.w_p = 0.5
                            # print(agent.w_p)
                            d_new = (da + dv) / norm(da + dv)
                            d = (d_new + agent.w_p * agent.g) / norm(d_new + agent.w_p * agent.g)
                        else:
                            d_new = (da + dv) / norm(da + dv)
                            d = d_new
                    else:
                        if i in self.leader_list:
                            agent.vel = agent.g * self.constant_speed

                    if norm(d) != 0:
                        angle_between = cal_angle_of_vector(d, agent.vel)
                        # logging.info("angle_between:{}".format(angle_between))
                        if angle_between >= self.theta_dot_max * self.dt:
                            # rotation_matrix_about 旋转后，返回的是向量
                            rot = rotation_matrix_about(agent.vel, self.theta_dot_max * self.dt)

                            vel0 = rot

                            rot1 = rotation_matrix_about(agent.vel, -self.theta_dot_max * self.dt)

                            vel1 = rot1

                            if cal_angle_of_vector(vel0, d) <= cal_angle_of_vector(vel1, d):
                                agent.vel = vel0 / norm(vel0) * self.constant_speed
                            else:
                                agent.vel = vel1 / norm(vel1) * self.constant_speed
                        else:
                            agent.vel = d * self.constant_speed

                    # 将邻居信息更新在obs_single中
                    # 将单个个体的观察空间长度固定
                    # 修改的地方在于加上本智能体的信息，在考虑与周边个体的关系时，同时需要本个体的位置和速度信息
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

                    obs_[i]  = obs_single
                # logging.info("obs_[i]:{},{}".format(i, obs_[i]))
            # 更新各个点的坐标位置
            # [agent.update_position(self.dt) for agent in self.swarm]

        total_velocity = 0

        
        unusual_flag = 0
        
        # 更新个体的位置
        for agent in self.swarm:
            
            # 不动
            if unusual_flag == 0:
                if agent.id not in failure_list:
                    agent.update_position(self.dt)
                    total_velocity = (agent.vel[0] * agent.g[0] + agent.vel[1] * agent.g[1]) / (self.constant_speed) + total_velocity
            
            
            # 随机运动
            # 异常个体生成一个随机运动方向
            if unusual_flag == 1:
                if agent.id in failure_list:
                    # 生成0到2pi内一个随机方向
                    random_direction = 2 * math.pi * random.uniform(0,1)
                    rand_x = math.cos(random_direction)
                    rand_y = math.sin(random_direction)
                    
                    d = np.array([rand_x, rand_y])
                    
                    angle_between_unusual = cal_angle_of_vector(d, agent.vel)
                        # logging.info("angle_between:{}".format(angle_between))
                    if angle_between_unusual >= self.theta_dot_max * self.dt:
                        # rotation_matrix_about 旋转后，返回的是向量
                        rot = rotation_matrix_about(agent.vel, self.theta_dot_max * self.dt)

                        vel0 = rot

                        rot1 = rotation_matrix_about(agent.vel, -self.theta_dot_max * self.dt)

                        vel1 = rot1

                        if cal_angle_of_vector(vel0, d) <= cal_angle_of_vector(vel1, d):
                            agent.vel = vel0 / norm(vel0) * self.constant_speed
                        else:
                            agent.vel = vel1 / norm(vel1) * self.constant_speed
                    else:
                        agent.vel = d * self.constant_speed
                
            
                agent.update_position(self.dt)
                total_velocity = (agent.vel[0] * agent.g[0] + agent.vel[1] * agent.g[1]) / (self.constant_speed) + total_velocity        
                    

            # 向无人集群相反方向运动
            if unusual_flag == 2:
                if agent.id in failure_list:
                    # 生成的方向为当前点与终点的反方向
                    x = self.target_x - agent.pos[0]
                    y = self.target_y - agent.pos[1]
                    
                    x_direction = x / math.sqrt(x**2 + y**2)
                    y_direction = y / math.sqrt(x**2 + y**2)
                    
                    d = np.array([-x_direction, -y_direction])
                    logging.warning("d:{}".format(d))
                    angle_between_unusual = cal_angle_of_vector(d, agent.vel)
                    # logging.warning("angle_between:{}".format(angle_between))
                    if angle_between_unusual >= self.theta_dot_max * self.dt:
                        # rotation_matrix_about 旋转后，返回的是向量
                        rot = rotation_matrix_about(agent.vel, self.theta_dot_max * self.dt)

                        vel0 = rot

                        rot1 = rotation_matrix_about(agent.vel, -self.theta_dot_max * self.dt)

                        vel1 = rot1

                        if cal_angle_of_vector(vel0, d) < cal_angle_of_vector(vel1, d):
                            agent.vel = vel0 / norm(vel0) * self.constant_speed
                        else:
                            agent.vel = vel1 / norm(vel1) * self.constant_speed
                    else:
                        agent.vel = d * self.constant_speed
                        
                    logging.warning("unusual:{}".format(agent.vel))
                    
                    print("failure")
                    
                agent.update_position(self.dt)
                total_velocity = (agent.vel[0] * agent.g[0] + agent.vel[1] * agent.g[1]) / (self.constant_speed) + total_velocity   
                
                
            
        # 在知情者方向的平均速度计算，正常个体的平均速度/总数量
        performance = total_velocity / (self.n) 


        with open("performance_curve_xu.txt","a+") as performance_curve:
            performance_curve.write(str(performance)+"\n")        

       
        # 稳态的时刻
        stable_moment = 0
        # 稳态的时刻
        # 计算所有节点不再与故障个体有交互的时刻
        if len(failure_list) !=0 and self.total_steps > 200:
            time_flag = True
            break_flag = False
            for item in self.max_sub_swarm:
                if item.id not in failure_list:
                    for id in failure_list:
                        if math.sqrt((item.pos[0] - self.swarm[id].pos[0]) ** 2 +
                                    (item.pos[1] - self.swarm[i].pos[1]) ** 2) < self.attract_range:
                            time_flag = False
                            break_flag = True
                            break
                if  break_flag == True:
                    break

            # 加一个逻辑，控制只写一次
            if time_flag == True and self.write_once_flag == True:
                with open("time_flag_step_xu.txt","w") as flag_step:
                    # logging.info("time_flag:{}".format(i))
                    flag_step.write(str(self.total_steps)+"\n") 
                self.write_once_flag = False 
                stable_moment = self.total_steps

        # 输出各个智能体的编号，坐标，速度方向,是否是领导者
        # logging.info("#########################")
        # for i in range(len(self.swarm)):
        #     logging.info("swarm:{},{},{}".format(self.swarm[i].id, self.swarm[i].pos, self.swarm[i].vel))

        if self.is_visual:
            # 可视化展示

            x = np.array([])
            y = np.array([])

            x_dot = np.array([])
            y_dot = np.array([])

            for agent in self.swarm:
                # 存储所有横纵坐标位置
                # x，y分别存储x,y
                x = np.append(x, agent.pos[0])
                y = np.append(y, agent.pos[1])

                # 存储所有横纵速度方向
                # x_dot，y_dot 分别存储x,y
                x_dot = np.append(x_dot, agent.vel[0])
                y_dot = np.append(y_dot, agent.vel[1])

            # 清除展示区
            self.ax.clear()
            # logging.info("x:{}".format(x))
            # logging.info("y:{}".format(y))
            # logging.info("x_dot:{}".format(x_dot))
            # logging.info("y_dot:{}".format(y_dot))
            # 设置箭头的形状和大小
            # 追随者
            x_temp = np.array([])
            y_temp = np.array([])
            x_temp_dot = np.array([])
            y_temp_dot = np.array([])

            # 领导者
            x_temp_f = np.array([])
            y_temp_f = np.array([])
            x_temp_dot_f = np.array([])
            y_temp_dot_f = np.array([])
            
            
            # 异常
            x_temp_fail = np.array([])
            y_temp_fail = np.array([])
            x_temp_dot_fail = np.array([])
            y_temp_dot_fail = np.array([])

            # logging.info("self.leader_list:{}".format(self.leader_list))
            for i in range(len(self.swarm)):
                if i not in list(self.leader_list) and i not in list(failure_list):
                    # x，y分别存储x,y方向上的位置
                    x_temp = np.append(x_temp, self.swarm[i].pos[0])
                    y_temp = np.append(y_temp, self.swarm[i].pos[1])

                    # x_dot，y_dot 分别存储x,y方向上的范数
                    x_temp_dot = np.append(x_temp_dot, self.swarm[i].vel[0] / norm(self.swarm[i].vel) * 0.4)
                    y_temp_dot = np.append(y_temp_dot, self.swarm[i].vel[1] / norm(self.swarm[i].vel) * 0.4)

            self.ax.quiver(x_temp, y_temp, x_temp_dot, y_temp_dot, width=0.01,
                           scale=5, units="inches", color='#EC3684', angles='xy')

            for item in self.leader_list:
                # x，y分别存储x,y方向上的位置

                x_temp_f = np.append(x_temp_f, self.swarm[item].pos[0])
                # logging.info("x_temp_f:{}".format(x_temp_f))
                y_temp_f = np.append(y_temp_f, self.swarm[item].pos[1])

                # x_dot，y_dot 分别存储x,y方向上的范数
                x_temp_dot_f = np.append(x_temp_dot_f, self.swarm[item].vel[0] / norm(self.swarm[item].vel) * 0.4)
                y_temp_dot_f = np.append(y_temp_dot_f, self.swarm[item].vel[1] / norm(self.swarm[item].vel) * 0.4)
            # logging.info("x_temp_f:{}".format(x_temp_f))

            self.ax.quiver(x_temp_f, y_temp_f, x_temp_dot_f,
                           y_temp_dot_f,
                           width=0.01, scale=5, units="inches", color='#006400', angles='xy')
            
            if len(failure_list)!=0: 
                # 给异常个体着色
                for item in failure_list:
                    # x，y分别存储x,y方向上的位置
                    logging.warning("item:{}".format(item))
                    x_temp_fail = np.append(x_temp_fail, self.swarm[item].pos[0])
                    # logging.info("x_temp_f:{}".format(x_temp_f))
                    y_temp_fail = np.append(y_temp_fail, self.swarm[item].pos[1])

                    # x_dot，y_dot 分别存储x,y方向上的范数
                    x_temp_dot_fail = np.append(x_temp_dot_fail, self.swarm[item].vel[0] / norm(self.swarm[item].vel) * 0.4)
                    y_temp_dot_fail = np.append(y_temp_dot_fail, self.swarm[item].vel[1] / norm(self.swarm[item].vel) * 0.4)
                # logging.info("x_temp_f:{}".format(x_temp_f))

                self.ax.quiver(x_temp_fail, y_temp_fail, x_temp_dot_fail,
                            y_temp_dot_fail,
                            width=0.01, scale=5, units="inches", color='#000000', angles='xy')
                
            
            

            # 添加画线, 画出排斥和吸引，判断是否正常运行
            for k in range(len(self.swarm)):
                # 画self.swarm[k] 与其邻居的线
                # 创建邻居集合
                neibors_attrack = []
                for m in range(len(self.swarm[k].neibour_set_attract)):
                    neibors_attrack.append(
                        [self.swarm[k].neibour_set_attract[m].pos[0], self.swarm[k].neibour_set_attract[m].pos[1]])
                if len(neibors_attrack) != 0:
                    x_points, y_points = zip(*neibors_attrack)
                    # 绘制连线
                    for x, y in neibors_attrack:
                        plt.plot([self.swarm[k].pos[0], x], [self.swarm[k].pos[1], y], 'g--', linewidth=0.1, zorder=1)
                # 排斥区域邻居集合
                neibors_repluse = []
                for m in range(len(self.swarm[k].neibour_set_repulse)):
                    neibors_repluse.append(
                        [self.swarm[k].neibour_set_repulse[m].pos[0], self.swarm[k].neibour_set_repulse[m].pos[1]])
                if len(neibors_repluse) != 0:
                    x_points, y_points = zip(*neibors_repluse)
                    # 绘制连线
                    for x, y in neibors_repluse:
                        plt.plot([self.swarm[k].pos[0], x], [self.swarm[k].pos[1], y], 'r--', linewidth=0.1, zorder=1)

            self.ax.set_aspect('auto', 'box')
            self.ax.set_xlim(0, field.width)
            self.ax.set_ylim(0, field.height)

            self.ax.tick_params(axis='x', colors='red')
            self.ax.tick_params(axis='y', colors='blue')
            circle = plt.Circle((self.target_x, self.target_y), self.target_radius, color='r', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.pause(0.01)
        # 清除个体的邻居集合
        for n in range(len(self.swarm)):
            self.swarm[n].neibour_set_attract = []
            self.swarm[n].neibour_set_repulse = []


        self.total_steps = self.total_steps + 1
        self.old_swarm = copy.deepcopy(self.swarm)

      
        # 奖励函数设计, observation的设计
        # 连通度设计奖励，到达奖励的设计
        # 连通度奖励就以连通度为奖励
        # 到达终点时的奖励设计
        done = [False] * self.n

        obs1_ = convert_list2(obs_)
        reward_temp = []  
        # 奖励函数1
        # 周边个体的数量奖励和到达目标的奖励
        for i in range(len(obs_)):
            num = 0
            for item1 in obs_[i]:              
                for item2 in item1:
                    if item2 != 0:
                        num = num + 1
                        break
            if num - 1 == 0:
                reward_temp.append(-1)
            else:
                reward_temp.append(num - 1)

        # 奖励2 所有故障个体与正常个体分离的时刻
        # 分离的时间越短，奖励越大    
        # for i in range(len(obs_)):
        #     # 离开不动个体的时间, 设计一个与离开不动个体时间相关的奖励,离开
        #     reward_resilience = 0
        #     if stable_moment == 0:
        #         reward_resilience = 0
        #     else:
        #         reward_resilience = 1/stable_moment * 3000
        #     reward_temp[i] = reward_temp[i] + reward_resilience

        return obs1_,  reward_temp, done,


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
        
        
        # 找到包含领导者的最大子群
        max_swarm_contain_leaders = []
        max_sub_size = 0
        for item in cons_flock:
            for sub_item in item:
                if sub_item.id in self.leader_list:
                    if len(item) > max_sub_size:
                        max_sub_size = len(item)
                        max_swarm_contain_leaders = item
                    break
        
        self.max_sub_swarm = max_swarm_contain_leaders   
        
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
            if (
                math.sqrt(
                    (point.pos[0] - self.target_x) ** 2
                    + (point.pos[1] - self.target_y) ** 2
                )
                < self.attract_range
            ):
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

def resilience_cal_display():
    # 从100步开始
    resilience_v = 0
    # 获取性能数据，
    data = []
    with open("performance_curve_xu.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            data.append(float(line))
    times_stable = 0
    with open("time_flag_step_xu.txt", "r", encoding="utf-8") as f:
        for line in f:
           times_stable =  int(line.strip("\n"))
    time_period_before_disturblance = 50
    # 故障从200,所以这里改成
    times_start = 200 - time_period_before_disturblance
    time_period = 100
    data_resilience = data[times_start : times_stable + time_period]
    data_resilience_sg = sg(data_resilience, window_length=25, polyorder=1)
    
    data_after_destroy = data_resilience_sg[times_stable - times_start : times_stable - times_start + time_period]
    # 韧性计算
    # y_d 期望性能
    id1, min_value = get_min(data_resilience_sg[time_period_before_disturblance:-time_period])
    # logging.info("id1:{}, min_value:{}".format(id1, min_value))
    y_d = 0
    for i in range(time_period_before_disturblance):
        y_d = y_d + data_resilience_sg[i]

    y_d = y_d / time_period_before_disturblance
    # y_r 恢复后性能
    y_r = 0
    # 获取恢复后的平均值和方差
    meanvalue_after_destroy = np.mean(data_after_destroy)

    # print("meanvalue_after_destroy:", meanvalue_after_destroy, " ", "var_after_destroy:", var_after_destroy)

    y_r = meanvalue_after_destroy



    # y_min 最低性能
    y_min = min_value
    # t_0 感兴趣时段起始时间
    t_0 = 0
    # t_d 遭受扰动时间
    t_d = time_period_before_disturblance
    # 开始恢复时间
    t_r = id1
    # 恢复到稳态的时间
    t_ss = times_stable
    # 感兴趣时段结束时间
    t_final = len(data_resilience_sg)

    # 最低性能要求


    #  总性能因子
    y_m = 0

    data_resilience_sg_sum = 0
    for item in data_resilience_sg:
        if item > y_m:
            data_resilience_sg_sum = data_resilience_sg_sum + item
    
    # 性能占比 
    sigma = data_resilience_sg_sum / (y_d * len(data_resilience_sg))



    #  rho 恢复因子
    rho = y_r / y_d

    #  最低性能因子
    delta = y_min / y_d

    # 恢复时间因子
    # t_ss - times_start 这个就是故障发生到稳态的时间，t_final - t_0是总时间
    tau = (t_ss - times_start) / (t_final - t_0)
    tau = 1

    # 波动因子
    zeta = calculate_fluctuation(data_resilience, data_resilience_sg)

    # 设置绝对时间尺度因子B, 将B设置为50
    delta_l = 0.8
    B = 300

    # 计算到稳态后较长的一段时间内，数据的平均值和方差

    # 做一个判断，比较恢复后的水平和最小的水平y_r和y_min
    # logging.info("y_r:{},y_min:{}".format(y_r, y_min))

    if y_r > y_min:
        if y_r < 0:
            y_r = 0
        if rho < 0:
            rho = 0 
        if sigma < 0 or sigma>=1:
            sigma = 0 
        if delta < 0:
            delta = 0
        resilience_v = rho * sigma * (delta + zeta) * (delta_l ** (len(data_resilience) / B))
        value = delta_l ** (len(data_resilience) / B)
        logging.warning("r:{},{},{},{},{}".format(rho,sigma,delta + zeta,value,resilience_v))
    else:
        resilience_v = 0

    with open("resilience_xu.txt", "a+", encoding="utf-8") as f:
        f.write(str(resilience_v)+"\n")


    plt.plot(data_resilience_sg, color="r", ls="--", label="smoothed")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("velocity")

    # 标出性能最低点和稳态时间点
    plt.annotate(
        "minimum_value",
        xy=(id1 + time_period_before_disturblance + 0.1, min_value),
        xytext=(id1 + time_period_before_disturblance + 0.1, min_value + 0.1),
        weight="bold",
        color="b",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
    )
    # plt.show()
    # 删除
    try:
        delete_file_if_exists("performance_curve_xu.txt")
        delete_file_if_exists("time_flag_step_xu.txt")
    except OSError as e:
        print(f"删除文件时发生错误: {e}")

    return resilience_v

def data_average(path):
    data = []
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip()!="":
                logging.info("line:{}".format(line))
                data.append(float(line.rstrip("\n")))
    cons = np.mean(data)   
    return cons




    



if __name__ == '__main__':
    import os
    file_path_space = "space_complexity_xu.txt"
    file_path_time = "time_complexity_xu.txt"
    file_path_connect = "connect_value_xu.txt"
    file_path_destination = "destination_nums_xu.txt"
    file_path_performance_curve = "performance_curve_xu.txt"
    file_path_time_flag = "time_flag_step_xu.txt"
    file_path_log = "test_log_0_xu.txt"
    file_path_resilience = "resilience_xu.txt"

    try:
        delete_file_if_exists(file_path_space)
        delete_file_if_exists(file_path_time)
        delete_file_if_exists(file_path_connect)
        delete_file_if_exists(file_path_destination)
        delete_file_if_exists(file_path_performance_curve)
        delete_file_if_exists(file_path_time_flag)
        delete_file_if_exists(file_path_log)
        delete_file_if_exists(file_path_resilience)
    except OSError as e:
        print(f"删除文件时发生错误: {e}")

    try:
        delete_file_if_exists("performance_curve_xu.txt")
        delete_file_if_exists("`time_flag_step_xu`.txt")
    except OSError as e:
        print(f"删除文件时发生错误: {e}")



    # import json
    # # 打开文件并加载JSON数据
    # with open('config.json', 'r') as file:
    #     config_data = json.load(file)
    # N = config_data["Num"]
    # P = config_data["P"]
    # # leader_list = config_data["Leader_list"]
    leader_list = [1,2]
    couzin = Couzin(5,0.2,True)
    couzin.attract_range = 30
    couzin.leader_list = leader_list
    print("leader_list",couzin.leader_list)


    leaders = couzin.leader_list
    w = 0.5
    angle = 2 * math.pi
    actions = []
    logging.info("runnning")
    for i in range(len(couzin.swarm)):
        if i in leaders:
            actions.append(w)
        else:
            actions.append(angle)
    score_history = []
    index_history = []
    couzin.talker_listener(actions)

    # for i in range(2000):
    #     couzin.step(actions=actions)


        
