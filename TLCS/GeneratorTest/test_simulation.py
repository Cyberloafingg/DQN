import traci
import numpy as np
import random
import timeit
from MyGenerator import TrafficGenerator
from MyGenerator import set_sumo
from MyGenerator import set_sumo_test
from DQN_Model import DQNModel
from MemoryPool import Memory
import math
import os
import torch
import sys
from sumolib import checkBinary

# 进车道的路网id 4*3 = 12
LANE = ['E2TL_0', 'E2TL_1', 'E2TL_2',
        'N2TL_0', 'N2TL_1', 'N2TL_2',
        'S2TL_0', 'S2TL_1', 'S2TL_2',
        'W2TL_0', 'W2TL_1', 'W2TL_2']
DELT = ["e2Detector_E2TL_0", "e2Detector_E2TL_1", "e2Detector_E2TL_2",
        "e2Detector_N2TL_0", "e2Detector_N2TL_1", "e2Detector_N2TL_2",
        "e2Detector_S2TL_0", "e2Detector_S2TL_1", "e2Detector_S2TL_2",
        "e2Detector_W2TL_0", "e2Detector_W2TL_1", "e2Detector_W2TL_2"]

os.environ['SUMO_HOME'] = "D:\Program\sumo-1.2.0"


# ####


class simulation:
    def __init__(self, Model,sumo_cmd, max_steps, traffic_generator, training_epochs, batch_size, gamma, device):
        self._Model = Model
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._traffic_generator = traffic_generator
        self._reward_store = []
        self._wait_time_store = []
        self._sum_lenth_store = []
        self._sum_brake_store = []
        self._training_epochs = training_epochs
        self._batch_size = batch_size
        self._gamma = gamma
        self._device = device

    def run(self, episode):
        start_time = timeit.default_timer()
        self._traffic_generator.generate_routefile(episode)
        self._sum_reward = 0
        self._sum_wait_time = 0
        self._sum_lenth = 0
        self._sum_brake = 0
        self._step = 0
        traci.start(self._sumo_cmd)
        last_phase_action = 0
        last_wait_time = 0
        traci.simulationStep()
        # 开始仿真
        while self._step < self._max_steps:
            # 计算当前状态 s_t+1
            current_state = self.get_state()
            # 计算当前奖励 r_t
            reward, last_wait_time,length_num, brake_num  = self.get_reward(last_wait_time)
            # 计算当前动作 a_t, phase_action返回0-7, yellow_action返回0-3
            phase_action, yellow_action = self.get_action(current_state)
            if self._step != 0 and last_phase_action != phase_action :
                self.set_yellow(last_phase_action, yellow_action)
            self.set_green(phase_action)
            last_phase_action = phase_action
            if reward < 0:
                self._sum_reward += reward
                self._sum_wait_time += last_wait_time
                self._sum_lenth += length_num
                self._sum_brake += brake_num
        traci.close()
        print(f"==================episode {episode}=======================")
        simulation_time = round(timeit.default_timer() - start_time, 1)
        print(f'SIMULATION simulation_time:{simulation_time}，sum_reward:{self._sum_reward},wait_time:{self._sum_wait_time},sum_lenth:{self._sum_lenth},sum_brake:{self._sum_brake}')
        # 保存数据
        self._reward_store.append(self._sum_reward)
        self._wait_time_store.append(self._sum_wait_time)
        self._sum_lenth_store.append(self._sum_lenth)
        self._sum_brake_store.append(self._sum_brake)

    def get_state(self):
        '''
        获取当前状态矩阵
        :return: 状态矩阵 1*12*50*2
        '''
        x = 12
        y = 50
        length = 5
        v_max = 13.89
        carbox = np.array(np.zeros((1, x, y, 2)), dtype="float32")
        for carID in traci.vehicle.getIDList():
            v_lane = traci.vehicle.getLaneID(carID)  # 获取车辆所在车道ID
            if v_lane in LANE:
                index_ = LANE.index(v_lane)  # [0->11]
                v_distance = traci.vehicle.getDistance(carID)  # 获取车道下标，得到矩阵的行所在位置
                # traci.lane.getLength(v_lane) 500m
                delt_distance = traci.lane.getLength(v_lane) - v_distance
                if delt_distance <= y * length:  # 在检测范围内
                    v_x = index_
                    v_y = math.floor(delt_distance / length)
                    carbox[0][v_x][v_y][0] = 1  # 位置矩阵
                    carbox[0][v_x][v_y][1] = traci.vehicle.getSpeed(carID) / v_max  # 速度归一化值
        return carbox

    def get_reward(self, last_wait_time):
        '''
        计算奖励
        :param decide: 当前相位
        :param action_decide: 下一相位
        :param last_wait_time: 上一时刻等待时间
        '''
        # 系数
        k1, k2, k3 = -0.2, -0.2, -1
        # 排队长度, 通过E2检测器获取
        length, wait_time_next, num = 0, 0, 0
        for i in range(len(LANE)):
            length += traci.lanearea.getJamLengthMeters(DELT[i])
            # 累计等待时间
            wait_time_next += traci.lane.getWaitingTime(LANE[i])
        r_wait_time = wait_time_next - last_wait_time
        # 刹车数量之和
        # print(traci.vehicle.getIDList())
        for car in traci.vehicle.getIDList():
            if traci.vehicle.getLaneID(car) in LANE and traci.vehicle.getAcceleration(car) < 0:
                num += 1
        reward = k1 * length + k2 * r_wait_time + k3 * num
        print(f'length:{length},r_wait_time :{wait_time_next},num:{num},reward:{reward}')
        # 平均排队长度：sum_length/记录的求reward次数
        # 平均等待时间：sum_wait_time/记录的求reward次数
        return reward, wait_time_next, length, num

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self._device)
        phase_action, yellow_action = self._Model.get_action(state)
        return phase_action, yellow_action

    def set_yellow(self, phase, yellow_action):
        yellow_phase_code = phase * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)
        self.simulate_step(yellow_action + 2)

    def set_green(self, phase):
        green_phase_code = phase * 2
        traci.trafficlight.setPhase("TL", green_phase_code)
        self.simulate_step(10)

    def simulate_step(self, step):
        if (self._step + step > self._max_steps):
            step = self._max_steps - self._step
        while step > 0:
            traci.simulationStep()
            step -= 1
            self._step += 1



total_episode = 2
batch_size = 200
train_epochs = 500
gamma = 0.9
max_steps, n_cars_generated = 5400, 1000
traffic_generator = TrafficGenerator(max_steps, n_cars_generated)
model = torch.load('model/model_name1680959135.2242284.pth')
sumo_cmd = set_sumo_test(gui=True, sumocfg_file_name='sumo_config.sumocfg', max_steps=max_steps)
simulation = simulation(model, sumo_cmd, max_steps, traffic_generator, train_epochs, batch_size, gamma,'cuda')

for episode in range(1, total_episode):
    simulation.run(episode)

import time
# reward, wait_time, brake_num, length
np_a = np.array([simulation._reward_store,simulation._wait_time_store,simulation._sum_brake_store,simulation._sum_lenth_store]).T
time_ = time.time()
np.savetxt(f"test{time_}.csv", np_a, delimiter=",")

