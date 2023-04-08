import numpy as np
import math

# 一共12条路线
route_info = ["W_N", "W_E", "W_S", "N_W", "N_E", "N_S", "E_W", "E_N", "E_S", "S_W", "S_N", "S_E"]
straight_line = ["W_E", "E_W", "N_S", "S_N"]
turn_line = ["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"]
right_turn_line = ["W_S", "N_W", "E_N","S_E"]
left_turn_line = ["W_N", "N_E",  "E_S", "S_W"]

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
    def generate_routefile(self, seed):
        """
        生成路网信息
        """
        np.random.seed(seed)  # make tests reproducible
        # 翻译：车辆的生成遵循威布尔分布,生成的车辆数目为self._n_cars_generated
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)
        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated
        # print(car_gen_steps)
        # produce the file for cars generation, one car per line
        with open(f"episode_routes.rou.xml", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>
            <routes>
            <vType accel="2.5" decel="4.5" id="standard_car" length="5.0" minGap="0.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)
            # car_counter是0-max的序号
            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                # route为路线，depart为出发时间，departLane为出发车道，departSpeed为出发速度
                if straight_or_turn < 0.5:  #0.6的概率是直行
                    route_straight = np.random.randint(0, 4)
                    for i in range(4):
                        if route_straight == i:
                            print(f"    <vehicle id=\"{straight_line[i]}_{car_counter}\" type=\"standard_car\" route=\"{straight_line[i]}\" depart=\"{step}\" departLane=\"random\" departSpeed=\"10\" />", file=routes)
                else:  # 0.4的概率是转弯
                    if straight_or_turn < 0.85:  # 0.35的概率是右转
                        route_turn = np.random.randint(0, 4)
                        for i in range(4):
                            if route_turn == i:
                                print(f"    <vehicle id=\"{right_turn_line[i]}_{car_counter}\" type=\"standard_car\" route=\"{right_turn_line[i]}\" depart=\"{step}\" departLane=\"random\" departSpeed=\"10\" />", file=routes)
                    else:  # 0.15的概率是左转
                        route_turn = np.random.randint(0, 4)
                        for i in range(4):
                            if route_turn == i:
                                print(f"    <vehicle id=\"{left_turn_line[i]}_{car_counter}\" type=\"standard_car\" route=\"{left_turn_line[i]}\" depart=\"{step}\" departLane=\"random\" departSpeed=\"10\" />", file=routes)

                    # route_turn = np.random.randint(0, 8)
                    # for i in range(8):
                    #     if route_turn == i:
                    #         print(f"    <vehicle id=\"{turn_line[i]}_{car_counter}\" type=\"standard_car\" route=\"{turn_line[i]}\" depart=\"{step}\" departLane=\"random\" departSpeed=\"10\" />", file=routes)
            print("</routes>", file=routes)


import traci
import os
import sys
from sumolib import checkBinary

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_name, "--no-step-log", "true",
                "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

# max_steps,n_cars_generated = 200,500
# TrafficGenerator = TrafficGenerator(max_steps, n_cars_generated)
# TrafficGenerator.generate_routefile(42)
# os.environ['SUMO_HOME'] = "D:\Program\sumo-1.2.0"
# sumo_cmd = set_sumo(gui=True, sumocfg_file_name='sumo_config.sumocfg', max_steps=max_steps)
# print(sumo_cmd)
# traci.start(sumo_cmd)
# step = 0
# # 进车道的路网id 4*3 = 12
# LANE = ['E2TL_0', 'E2TL_1', 'E2TL_2',
#         'N2TL_0', 'N2TL_1', 'N2TL_2',
#         'S2TL_0', 'S2TL_1', 'S2TL_2',
#         'W2TL_0', 'W2TL_1', 'W2TL_2']
# DELT = ["e2Detector_E2TL_0", "e2Detector_E2TL_1", "e2Detector_E2TL_2",
#         "e2Detector_N2TL_0", "e2Detector_N2TL_1", "e2Detector_N2TL_2",
#         "e2Detector_S2TL_0", "e2Detector_S2TL_1", "e2Detector_S2TL_2",
#         "e2Detector_W2TL_0", "e2Detector_W2TL_1", "e2Detector_W2TL_2"]
#
# y = 50 # 50 * 5 = 250m
# x = 12
# length = 5
# v_max = 13.89
#
# while step < max_steps:
#     carbox = np.array(np.zeros((1, x, y, 2)), dtype="float32")
#     for carID in traci.vehicle.getIDList():
#         v_lane = traci.vehicle.getLaneID(carID) # 获取车辆所在车道ID
#         if v_lane in LANE:
#             index_ = LANE.index(v_lane) #[0->11]
#             v_distance = traci.vehicle.getDistance(carID)  # 获取车道下标，得到矩阵的行所在位置
#             # traci.lane.getLength(v_lane) 500m
#             delt_distance = traci.lane.getLength(v_lane) - v_distance
#             if delt_distance <= y * length:  # 在检测范围内
#                 v_x = index_
#                 v_y = math.floor(delt_distance / length)
#                 carbox[0][v_x][v_y][0] = 1  # 位置矩阵
#                 carbox[0][v_x][v_y][1] = traci.vehicle.getSpeed(carID) / v_max  # 速度归一化值
#
#     # 计算累计等待时间
#     # 排队长度, 通过E2检测器获取
#     length_, wait_time_next, r_phase, num = 0, 0, 0, 0
#     for i in range(len(LANE)):
#         length_ += traci.lanearea.getJamLengthMeters(DELT[i])
#         # 累计等待时间
#         wait_time_next += traci.lane.getWaitingTime(LANE[i])
#     # r_wait_time = wait_time_next - wait_time
#     light = traci.trafficlight.getPhase("TL");
#     traci.trafficlight.setPhaseDuration("TL", 5)
#     print(light)
#     # 计算刹车次数
#     for car in traci.vehicle.getIDList():
#         if traci.vehicle.getLaneID(car) in LANE and traci.vehicle.getAcceleration(car) < 0:
#             num += 1
#     Time = traci.simulation.getTime()
#     print(f"{Time} : 累计等待时间：{wait_time_next}，排队长度：{length_}，刹车次数：{num}")
#     traci.simulationStep()
#     step += 1
# traci.close()