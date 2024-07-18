import math
import os
import shutil


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter as sg
import logging
# logging.basicConfig(
#     level=logging.INFO,  # 控制台打印的日志级别
#     filename="test.txt",
#     filemode="w",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#     # a是追加模式，默认如果不写的话，就是追加模式
#     format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
#     # 日志格式
# )

# logger = logging.getLogger()
# logger.setLevel(logging.WARNING)



def calculate(a1, a2):
    """
    Parameters
    ----------
    a1 : list
        横坐标，因为时间轴不一定是像0,1,2,3,4,5
    a2 : list
        纵坐标，y值

    Returns
    -------
    a1[min_id] : list
        min_id对应的index值, a1[min_id]对应稳态时间值
    min_var : float
        对应的方差
    """
    min_id = 0
    min_var = float("Inf")

    for i in range(0, len(a2) - 1):
        # logging.info("i:%s", str(i))
        var = 0
        total_sum = 0
        for j in range(i, len(a2)):
            total_sum = total_sum + a2[j]
        mean = total_sum / (len(a2) - i)
        for j in range(i, len(a2)):
            var = var + pow((a2[j] - mean), 2)
        logging.info(
            "i:%s,var:%s,num:%s,value:%s", str(i), str(var), str(pow(len(a2) - i, 2)), str(var / pow(len(a2) - i, 2))
        )
        var = var / pow(len(a2) - i, 2)

        if var < min_var:
            min_var = var
            min_id = i
        if min_var < 0.0001:
            return a1[min_id], min_var
    return a1[min_id], min_var


def time_amend(real_time, absolute_time):
    omega = 0.6
    time_factor = math.pow(omega, real_time / absolute_time)
    return time_factor





# 增加一个功能，对最低需求满足程度
def get_min(data):
    # 获取最小值及最小值下标
    min_value = float("inf")
    id1 = 0
    for i in range(0, len(data)):
        if data[i] < min_value:
            min_value = data[i]
            id1 = i
    return id1, min_value


def calculate_fluctuation(smooth_data_before, smooth_data_after):
    """
    此函数主要作用为计算波动因子。

    Args:
        smooth_data_before(list) : 滤波前的数据
        smooth_data_after(list) : 滤波后的数据
    Return:
        波动因子
    """
    p_s = 0
    # 平滑前和平滑后数据差的平方和
    p_n = 0
    for item in smooth_data_after:
        p_s = p_s + item**2
    for i in range(len(smooth_data_after)):
        p_n = p_n + (smooth_data_before[i] - smooth_data_after[i]) ** 2
    snr_db = 10 * math.log((p_s / p_n), 10)
    zeta = 1 / (1 + math.exp(-0.25 * (snr_db - 15)))
    return zeta


# data = []
# # 性能数据
# for line in open("shape_data.txt", "r", encoding="utf-8"):
#     line = line.strip("\n")
#     data.append(float(line))

# times_stable = 0

# # 判断集群在仿真结束时是否裂开，如果未裂开，则将韧性设为0
# size = 0
# subnetwork_num = 0
# lines_in_data = 0
# with open("data.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     lines_in_data = len(lines)
#     if lines_in_data == 2:
#         # 到稳定时刻子网络的个数
#         subnetwork_size = lines[0].strip("\n")
#         # 稳定时刻
#         times_stable = lines[1].strip("\n")

# data_resilience = []
# data_resilience_sg = []
# time_period_before_disturblance = 0

# # 韧性值初始为0
# r = 0
# r_tran = 0
# r_liu = 0
# r_cheng = 0

# id1 = 0
# if lines_in_data == 2:
#     times_stable = int(float(times_stable))

#     time_period_before_disturblance = 100

#     times_start = 750 - time_period_before_disturblance

#     time_period = 300

#     # 获取的时间段为故障前(time_period_before_disturblance)个step到稳态后的(time_period)个时间段
#     data_resilience = data[times_start : times_stable + time_period]

#     # sg滤波 window_length: 窗口长度，该值为正奇数，polyorder越大越平滑
#     # window_length值越小，则曲线越贴近真实曲线
#     # polyorder: 用于拟合样本的多项式的阶数
#     data_resilience_sg = sg(data_resilience, window_length=25, polyorder=1)

#     # 故障发生后，稳态之后，的一段period的时间，主要是为了计算恢复后的均值，就是恢复后的水平
#     data_after_destroy = data_resilience_sg[times_stable - times_start : times_stable - times_start + time_period]

#     # 获取最小值，这个最小值的区间为故障发生到稳态时刻，区间就是故障开始到故障结束
#     id1, min_value = get_min(data_resilience_sg[time_period_before_disturblance:-time_period])
#     # 这个id就是故障发生的时刻, data_resilience_sg的长度就是时间的总长，要计算的几个数据

#     logging.info("id1:{}".format(id1))

#     # if min_value < 0:
#     #     min_value = 0

#     # print("times_stable:", times_stable - times_start)
#     # print("id1:", id1 + 50, " ", "min_value:", min_value)

#     # 韧性计算
#     # y_d 期望性能
#     y_d = 0
#     for i in range(time_period_before_disturblance):
#         y_d = y_d + data_resilience_sg[i]

#     y_d = y_d / time_period_before_disturblance

#     # y_r 恢复后性能
#     y_r = 0
#     # print(len(data_resilience_sg))

#     # 获取恢复后的平均值和方差
#     meanvalue_after_destroy = np.mean(data_after_destroy)
#     var_after_destroy = np.var(data_after_destroy)

#     # print("meanvalue_after_destroy:", meanvalue_after_destroy, " ", "var_after_destroy:", var_after_destroy)

#     y_r = meanvalue_after_destroy



#     # y_min 最低性能
#     y_min = min_value
#     # t_0 感兴趣时段起始时间
#     t_0 = 0
#     # t_d 遭受扰动时间
#     t_d = time_period_before_disturblance
#     # 开始恢复时间
#     t_r = id1
#     # 恢复到稳态的时间
#     t_ss = times_stable
#     logging.info("t_ss:{}".format(t_ss))
#     # 感兴趣时段结束时间
#     t_final = len(data_resilience_sg)

#     # 最低性能要求


#     #  总性能因子
#     sigma0 = sum(data_resilience_sg) / (y_d * len(data_resilience_sg))
#     y_m = 6

#     data_resilience_sg_sum = 0
#     for item in data_resilience_sg:
#         if item > y_m:
#             data_resilience_sg_sum = data_resilience_sg_sum + item
#     sigma = data_resilience_sg_sum / (y_d * len(data_resilience_sg))



#     #  rho 恢复因子
#     rho = y_r / y_d

#     #  最低性能银子
#     delta = y_min / y_d

#     # 恢复时间因子
#     # t_ss - times_start 这个就是故障发生到稳态的时间，t_final - t_0是总时间
#     tau = (t_ss - times_start) / (t_final - t_0)
#     tau = 1

#     # 波动因子
#     zeta = calculate_fluctuation(data_resilience, data_resilience_sg)

#     # 设置绝对时间尺度因子B, 将B设置为50
#     delta_l = 0.8
#     B = 300

#     # 计算到稳态后较长的一段时间内，数据的平均值和方差

#     # 做一个判断，比较恢复后的水平和最小的水平y_r和y_min
#     logging.info("y_r:{},y_min:{}".format(y_r, y_min))
#     if y_r > y_min:
#         r = rho * sigma * (delta + zeta) * (delta_l ** (len(data_resilience) / B))
#         logging.info("r:{}".format(r))
#     else:
#         r = 0

#     # Tran方法 = 面积占比 *恢复水平*(1 + 最低水平 + 波动因子 - 稳态时间**(最高水平 - 最低水平))
#     ""
#     r_tran = (
#         sigma0
#         * rho
#         * (1 + delta + zeta - ((times_stable - times_start) / (times_stable - times_start)) ** (rho - delta))
#     )

#     # Liu方法
#     r_liu = sigma

#     # Cheng方法
#     # Cheng方法 = 抗毁系数 * 面积占比 * 最低水平 + 恢复系数 * 面积占比 * 恢复水平
#     # 抗毁阶段survivability_sigma，恢复阶段recover_sigma
#     # 抗毁总时间为 id1 - 100, 恢复阶段总时间 len(data_resilience_sg) - id1 - 300
#     # 因其是离散点，因此直接求和即可
#     Delta_d = 0
#     Delta_r = 0

#     delta_cheng = 0.8
#     B_cheng = 300

#     Resistance_coefficient = 0.5
#     Recovery_cofficient = 0.5

#     Delta_d_num = 0
#     Delta_r_num = 0

#     logging.info("id1:{}".format(id1))
#     for i in range(len(data_resilience_sg)):
#         if i >= 100 and i <= id1 +100:
#             Delta_d = Delta_d + data_resilience_sg[i]
#             Delta_d_num = Delta_d_num + 1
#         if i >= (id1 +100) and i <= (len(data_resilience_sg) - 300):
#             Delta_r = Delta_r + data_resilience_sg[i]
#             Delta_r_num = Delta_r_num + 1

#     time_sensitivity_d = delta_cheng ** (B_cheng / (id1))
#     time_sensitivity_r = delta_cheng ** ((len(data_resilience_sg) - id1 - 300 - 100) / B_cheng)


#     # print(
#     #     Resistance_coefficient,
#     #     Delta_d / (Delta_d_num * y_d),
#     #     delta,
#     #     time_sensitivity_d,
#     #     Recovery_cofficient,
#     #     Delta_r / (Delta_r_num * y_d),
#     #     rho,
#     #     time_sensitivity_r,
#     # )


#     r_cheng = (
#         Resistance_coefficient * Delta_d / (Delta_d_num * y_d) * delta * time_sensitivity_d
#         + Recovery_cofficient * Delta_r / (Delta_r_num * y_d) * rho * time_sensitivity_r
#     )

#     # print(
#     #     "rho:",
#     #     rho,
#     #     " sigma:",
#     #     sigma,
#     #     " delta:",
#     #     delta,
#     #     " zeta:",
#     #     zeta,
#     #     " tau:",
#     #     tau,
#     #     " rho:",
#     #     rho,
#     #     " delta:",
#     #     delta,
#     #     "time_factor:",
#     #     delta_l ** (len(data_resilience) / B),
#     # )


#     def copy_file_to_media_folder():
#         current_folder = os.getcwd()  # 获取当前文件夹路径
#         source_file_data= os.path.join(current_folder, 'data.txt')  # 源文件路径
#         source_file_data_shape = os.path.join(current_folder, 'shape_data.txt')  # 源文件路径

#         media_folder = os.path.join(current_folder, 'media')  # 目标文件夹路径

#         # 检查目标文件夹是否存在，如果不存在则创建
#         if not os.path.exists(media_folder):
#             os.makedirs(media_folder)

#         # 复制文件到目标文件夹
#         destination_file = os.path.join(media_folder, 'data.txt')
#         shutil.copy(source_file_data, destination_file)

#         destination_file_shape = os.path.join(media_folder, 'shape_data.txt')
#         shutil.copy(source_file_data_shape, destination_file_shape)

#         print("文件已成功复制到'{}'文件夹下。".format(media_folder))


# # f = open("resilence.txt", "a+")
# # f.write(str(r_liu) + "\n")
# # f.close()

# print("###################################")

# print("process_time",times_stable - 750)
# print("r:",r)
# print("r_tran:", r_tran)
# print("r_cheng:", r_cheng)
# print("r_liu:", r_liu)


# plt.plot(data_resilience, label="raw")
# plt.plot(data_resilience_sg, color="r", ls="--", label="smoothed")
# plt.legend()
# plt.xlabel("steps")
# plt.ylabel("velocity")



################################################

# 图中标注功能
# 标出故障发生的时刻，因为data_resilience是从故障前100算的，所以此种计算方式没错
# plt.annotate(
#     "failure start ",
#     xy=(time_period_before_disturblance + 2, data_resilience_sg[time_period_before_disturblance]),
#     xytext=(time_period_before_disturblance + 2, data_resilience_sg[time_period_before_disturblance] + 1),
#     weight="bold",
#     color="b",
#     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
# )

# # 标出性能最低点和稳态时间点
# plt.annotate(
#     "minimum_value",
#     xy=(id1 + time_period_before_disturblance + 2, min_value),
#     xytext=(id1 + time_period_before_disturblance + 2, min_value + 1),
#     weight="bold",
#     color="b",
#     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
# )
# plt.annotate(
#     "stable_value",
#     xy=(times_stable - times_start, data_resilience_sg[times_stable - times_start]),
#     xytext=(times_stable - times_start, data_resilience_sg[times_stable - times_start] - 0.5),
#     weight="bold",
#     color="b",
#     arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="arc3", color="red"),
# )
# # 设置最低水平，蓝色线表示稳定值，红色线条表示稳定值
# plt.axhline(y=min_value, c="r", ls="-.", lw=1)
# plt.axhline(y=y_r, c="b", ls="-.", lw=1)


# plt.show()
# # plt.pause(0.1)
# plt.title("Interference data")