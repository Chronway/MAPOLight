# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 18:28:44 2023

@author: Wang Chong
"""

import os, sys
from typing import Callable, Optional, Tuple, Union

from pettingzoo.utils.env import AgentID, ObsType, ActionType

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# LIBSUMO 'LIBSUMO_AS_TRACI'如果在环境变量里有非空值为true，否则为false

from gymnasium import Env
# from gym.spaces.box import Box
import numpy as np
import pandas as pd
import sumolib
import traci
# from supersuit import pad_action_space_v0, pad_observations_v0

from utils.MobileSensor import CAVs
# from gym.envs.registration import EnvSpec
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .TrafficSignal import TrafficSignal
from .networkdata import NetworkData

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


# LIBSUMO = True


def env(**kwargs):
    env = SumoEnvironmentPZ(**kwargs)
    # print(env.observation_spaces)
    # env = pad_action_space_v0(env)#action和obs顺序可变
    # = pad_observations_v0(env) #可能有bug
    # env = pad_action_space_v0(env) #可能有bug
    # print(env.observation_spaces)
    # print("+++++++++++++++++++++++++++++++++++++")
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(Env):  # (MultiAgentEnv):
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
            self,
            config_file,
            net_file: str,
            route_file: str,
            # phases: dict = dict(),
            # lights: dict = dict(),
            PR=0.05,
            out_csv_name: Optional[str] = None,
            use_gui: bool = False,
            begin_time: int = 0,  # 可能只能设为0否则会出错
            num_seconds: int = 20000,
            max_depart_delay: int = 100000,
            time_to_teleport: int = -1,
            # time_to_load_vehicles=0,
            delta_time: int = 5,
            yellow_time: int = 3,
            min_green: int = 5,  # min_green == delta_time比较好，可以保证时间比较充足。
            max_green: int = 50,
            single_agent: bool = False,
            reward_fn: Union[str, Callable] = 'diff-waiting-time',
            sumo_seed: Union[str, int] = 'random',
            sumo_warnings: bool = True,
            cav_env: bool = False,  # whether cav env is activated

            # be careful：cav compare will consume alot of resouces and make the training intolerable long!!!
            cav_compare: bool = False,  # compare cav with all observe

            collaborate: bool = False,  # cooperative traffic signal,share reward and observations
    ):
        super(SumoEnvironment, self).__init__()  # ※bug:没有这句执行不下去，会报worker died error，还去网站问了

        self._net = net_file
        self._route = route_file
        self._cfg = config_file
        self.use_gui = use_gui
        self.sumo = None

        self.nd = NetworkData(self._net)
        self.netdata = self.nd.get_net_data()

        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        # self.phases = phases
        self.sim_max_time = num_seconds
        # self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts #后面可以统一为begin time
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle

        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        self.cav_env = cav_env
        self.cav_compare = cav_compare

        # assert not(not self.cav_env and self.cav_compare), "compare with full oberserves while cav enviroment is not activated!"
        if (not self.cav_env and self.cav_compare):
            print("compare with full oberserves while cav enviroment is not activated!")
            self.cav_compare == False

        if (self.cav_env):
            print("########################################")
            print("####### CAV environment started!!! #####")
            print("########################################")
            self.cav_set = CAVs(PR, net_file)  # penalty rate = 0.05 检查bug，去掉cav 2022.1.31
        else:
            print("########################################")
            print("#########  NO CAV environment!!! #######")
            print("########################################")
            self.cav_set = None

        if (self.cav_compare):
            print("########################################")
            print("######### CAV compare started!!! #######")
            print("########################################")
        else:
            print("########################################")
            print("#########  NO CAV compare!!! ###########")
            print("########################################")

        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings

        # self.lights_map = lights
        self.collaborate = collaborate

        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1

        if LIBSUMO:
            traci.start(
                [sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection' + self.label)
            conn = traci.getConnection('init_connection' + self.label)

        # idlist:['e3', 'e4', 'e7', 'e8']
        self.ts_ids = list(conn.trafficlight.getIDList())

        # print("ids:",self.ts_ids)
        # ids: ['A1', 'A2', 'A3', 'B0', 'B1', 'B2', 'B3', 'B4', 'C0', 'C1', 'C2', 'C3', 'C4', 'D0', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3']

        #####################size wrapper:11.5##################
        # self.max_obs = (41,)
        # self.max_action = 4
        ########################################################
        self.traffic_signals = {ts: TrafficSignal(self,
                                                  ts,
                                                  self.delta_time,
                                                  self.yellow_time,
                                                  self.min_green,
                                                  self.max_green,
                                                  # self.phases,
                                                  self.begin_time,
                                                  self.reward_fn,
                                                  conn) for ts in self.ts_ids}
        conn.close()

        self.vehicles = dict()  # 好像有什么用，删了报错
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        # self.spec = EnvSpec('SUMORL-v0') #没用到？

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.last_measure = {ts: None for ts in self.ts_ids}
        self.last_reward = {ts: None for ts in self.ts_ids}  # 不要 2022.9.9

        if (self.cav_compare):
            # self.last_diff = {ts: {'mean_diff':self.traffic_signals[ts].mean_diff,'std_diff':self.traffic_signals[ts].std_diff} for ts in self.ts_ids} #记录cav环境和全科观测环境的差异
            self.last_diff = {ts: {'correlation': self.traffic_signals[ts].correlation,
                                   'diffreward': self.traffic_signals[ts].diff_reward} for ts in
                              self.ts_ids}  # 记录cav环境和全科观测环境的差异

        # 为了组合reward
        if (self.collaborate):
            self.neignbors = dict()
            self.net = sumolib.net.readNet(net_file)
            self.getNeignbors()
            print(self.neignbors)

        # raise NameError

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                    # '-c', self._cfg,
                    '-n', self._net,
                    '-r', self._route,
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--waiting-time-memory', '10000',
                    '--time-to-teleport', str(self.time_to_teleport)]
        # if self.begin_time > 0:
        # sumo_cmd.append('-b {}'.format(self.begin_time)) #暂时不要，否则和自己设的冲突
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    '''
    调用方式：
    (应该是有其他地方调用)
    '''

    def reset(self, seed: Optional[int] = None, return_info=False, **kwargs):
        # print("ding wei env reset");
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []
        # self.cav_set.clear_cavs()#  感觉不需要了

        if seed is not None:
            self.sumo_seed = seed

        self._start_simulation()

        if self.run == 0:
            self.ts_ids = list(self.sumo.trafficlight.getIDList())
            # self.last_reward = {ts: 0 for ts in self.ts_ids}

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green,
                                                     self.max_green, self.begin_time, self.reward_fn, self.sumo)
            self.last_measure[ts] = 0.0

            if (self.cav_compare):
                # self.last_diff[ts] = {'mean_diff':self.traffic_signals[ts].mean_diff,'std_diff':self.traffic_signals[ts].std_diff}
                self.last_diff = {ts: {'correlation': self.traffic_signals[ts].correlation,
                                       'diffreward': self.traffic_signals[ts].diff_reward} for ts in
                                  self.ts_ids}  # 记录cav环境和全科观测环境的差异
                self.traffic_signals[ts].last_measure_total = 0

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.begin_time):  # 先加一点车,否则报NoneType错误
            self._sumo_step()

        if self.single_agent:
            if return_info:
                return self._compute_observations()[self.ts_ids[0]], self._compute_info()
            else:
                return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    '''
    step->self._compute_observations()
    step->self._compute_rewards()
    step->self._compute_step_info()
    step->self._sumo_step()
    '''

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            pass
        else:
            self._apply_actions(action)  # next phase
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()  # 更新绿灯phase

        # self._sumo_step()
        # 如果此处是self._sumo_step()，不能保证time_to_act一定是true。
        # 因此_compute_observations和self._compute_rewards(函数返回值仅可用于测试，不可用于传递obs和reward值。
        self._run_steps()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        # 注：observation和reward可测试当前step是否update了数据。
        # 方法为使用self._sumo_step()，然后有：
        # (1)如果update了数据，即next_action_time = sim_step，则有len(obs)和len(reward)=agent number
        # (2)否则，len(obs)和len(reward)=0

        # print(len(observation))
        # print("step reward:",reward)

        done = {'__all__': self.sim_step > self.sim_max_time}
        self._compute_lastmeasures()
        info = self._compute_info()
        # self.metrics.append(info)
        self.last_reward = reward
        

        if self.single_agent:
            return self.observations[self.ts_ids[0]], self.rewards[self.ts_ids[0]], done['__all__'], done['__all__'], {}
        else:
            return self.observations, self.rewards, done, done, info

    def _run_steps(self):
        # step = 0 #for test
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            # step += 1 #for test
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()  # update以后就可能切到新相位了
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True
        # print("sim tt steps:",step) #for test

    # action dict:{tsid1:action, tsid2:action...}
    def _apply_actions(self, actions):
        # print("ding wei _apply_actions");
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase_random(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase_random(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones

    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)
        return info

    def _compute_observations(self):
        # print("ding wei _compute_observations")
        # self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        computed = False  ####此处是否要测试一下observation是否有不同再决定这个flag要不要？☆☆☆☆潜在bug
        for ts in self.ts_ids:
            if self.traffic_signals[ts].time_to_act:
                if (self.cav_env and (not computed)):  # compute: only count once is enough
                    self.cav_set.update_detects()
                    computed = True
                self.observations.update({ts: self.traffic_signals[ts].compute_observation()})

        # self.observation_size_wrapper();
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if
                self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        # print("ding wei _compute_rewards");
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if
                             self.traffic_signals[ts].time_to_act})

        # considered adjacent rewards
        if (self.collaborate):
            for ts in self.ts_ids:
                if self.traffic_signals[ts].time_to_act:
                    self.rewards[ts] = self.mixed_reward(ts)

        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_lastmeasures(self):
        self.last_measure.update({ts: self.traffic_signals[ts].compute_lastmeasure() for ts in self.ts_ids})
        # return {ts: self.last_measure[ts] for ts in self.last_measure.keys()}

    '''wrapper for pettingzooenv
    def smaller_size(self,observation):
        ob_size,max_size = 1,1
        ob_shape = np.shape(observation)
        size_shape = np.shape(self.max_obs)
        
        assert len(ob_shape)==len(size_shape),"dim size not equal!"

        for dim in ob_shape:
            ob_size = ob_size*dim

        for dim in self.max_obs:
            max_size = max_size*dim

        if(ob_size<max_size): # smaller will add size wrapper
            #print("is smaller!!!",ob_shape)
            return True
        elif(ob_size>max_size):
            self.max_size=ob_shape
        return False
    
    #important: agents should have same obs space size:
    def observation_size_wrapper(self):
        self.observations.update({ts:self.change_obshape(self.observations[ts]) for ts in self.ts_ids if(self.smaller_size(self.observations[ts]))})

    #high dimension state (>1 dim) not implement
    def change_obshape(self,observation:np.ndarray):
        #@TODO:high dim implementation
        #if(len(np.shape(observation))>1):
            #np.reshape(observation,1)
        #高维思路：转成一维矩阵然后再转回高维矩阵
        #print(np.shape(observation))
        #print(np.shape(np.zeros(self.max_obs[0]-np.shape(observation)[0])))
        #print(np.shape(np.concatenate((observation,np.zeros(self.max_obs[0]-np.shape(observation)[0])),axis = 0)))
        #print("############")
        return np.concatenate((observation,np.zeros(self.max_obs[0]-np.shape(observation)[0])),axis = 0)
    '''

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()
        if (self.cav_env):  # 定义了车联网环境再跑
            self.cav_set.update_observations()  # 更新三集合：车集合，cav车集合，cav探测到的车集合

    def _compute_step_info(self):
        # for ts in self.ts_ids:
        #    print(self.last_diff[ts]['diffreward'])
        return {
            'step_time': self.sim_step,
            'reward': self.last_reward[self.ts_ids[0]],
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(self.last_measure[ts] for ts in self.ts_ids),
            # 'avg_wait_time': sum(self.last_measure[ts] for ts in self.ts_ids) / len(self.sumo.vehicle.getIDList()), #可能除0
            'avg_wait_time': np.mean([self.last_measure[ts] for ts in self.ts_ids]),
            'avg_speed': np.mean([self.sumo.vehicle.getSpeed(vid) for vid in self.sumo.vehicle.getIDList()]),

            # 'mean_diff_obs':np.mean([self.last_diff[ts]['mean_diff']['obs'] for ts in self.ts_ids]),#比较cav环境下和全观测环境下的obs和reward(waits)的差值
            # 'mean_diff_wait':np.mean([self.last_diff[ts]['mean_diff']['waits'] for ts in self.ts_ids]),
            # 'std_diff_obs':np.mean([self.last_diff[ts]['std_diff']['obs'] for ts in self.ts_ids]),#比较cav环境下和全观测环境下的obs和reward(waits)的std差值
            # 'std_diff_wait':np.mean([self.last_diff[ts]['std_diff']['waits'] for ts in self.ts_ids]),
            'correlation_obs': np.nanmean([self.last_diff[ts]['correlation']['obs'] for ts in
                                           self.ts_ids]) if self.cav_compare else 'COMPARE OFF',
            # 比较cav环境下和全观测环境下的obs和reward(waits)的std差值
            'correlation_wait': np.nanmean([self.last_diff[ts]['correlation']['waits'] for ts in
                                            self.ts_ids]) if self.cav_compare else 'COMPARE OFF',
            'diff_reward': np.mean([self.last_diff[ts]['diffreward']['result'] for ts in
                                    self.ts_ids]) if self.cav_compare else 'COMPARE OFF',

            'total_stop_signal1': self.traffic_signals[self.ts_ids[0]].get_total_queued(),
            'total_stop_signal2': self.traffic_signals[self.ts_ids[1]].get_total_queued(),
            'total_stop_signal3': self.traffic_signals[self.ts_ids[2]].get_total_queued(),
            'total_stop_signal4': self.traffic_signals[self.ts_ids[3]].get_total_queued(),

            'total_wait_time_signal1': self.last_measure[self.ts_ids[0]],
            'total_wait_time_signal2': self.last_measure[self.ts_ids[1]],
            'total_wait_time_signal3': self.last_measure[self.ts_ids[2]],
            'total_wait_time_signal4': self.last_measure[self.ts_ids[3]],

            'total_energy_consumption_signal1': np.sum(
                self.traffic_signals[self.ts_ids[0]].get_energy_consumption_per_lane()),
            'total_energy_consumption_signal2': np.sum(
                self.traffic_signals[self.ts_ids[1]].get_energy_consumption_per_lane()),
            'total_energy_consumption_signal3': np.sum(
                self.traffic_signals[self.ts_ids[2]].get_energy_consumption_per_lane()),
            'total_energy_consumption_signal4': np.sum(
                self.traffic_signals[self.ts_ids[3]].get_energy_consumption_per_lane()),

            'avg_speed_signal1': self.traffic_signals[self.ts_ids[0]].get_avg_speed(),
            'avg_speed_signal2': self.traffic_signals[self.ts_ids[1]].get_avg_speed(),
            'avg_speed_signal3': self.traffic_signals[self.ts_ids[2]].get_avg_speed(),
            'avg_speed_signal4': self.traffic_signals[self.ts_ids[3]].get_avg_speed()
        }

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        #        if self.disp is not None:
        #            self.disp.stop()
        #            self.disp = None
        self.sumo = None

    def __del__(self):
        self.close()

    def save_csv(self, out_csv_name, run):
        # print("ding wei save_csv");
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            print("save csv to:", out_csv_name + '_run{}'.format(run) + '.csv')
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    def getNeignbors(self):
        nodes = self.net.getNodes()
        for node in nodes:
            if (node.getID() in self.ts_ids):
                # print("---",node.getID())
                neignbors = node.getNeighboringNodes()
                neignborset = set()
                for neignbor in neignbors:
                    if (neignbor.getID() in self.ts_ids):
                        neignborset.add(neignbor.getID())
                self.neignbors.update({node.getID(): neignborset})

    def mixed_reward(self, mytid: str):
        center_weight = 0.5  # center weight = 0.5
        neignbor_tids = self.neignbors[mytid]
        neirewards = dict()  # neighnbor reward
        for tid in self.rewards.keys():
            if (tid in neignbor_tids):
                neirewards.update({tid: self.rewards[tid]})
        if len(neirewards) != 0:
            neignbor_weight = (1 - center_weight) / len(neirewards)
        else:
            neignbor_weight = 0
        # weighted_reward = center_weight * self.rewards[tid]
        weighted_reward = center_weight * self.rewards[mytid]

        # print("tid:",mytid,"neignbors:",neignbor_tids,"   neignbor weight:",neignbor_weight)
        # print("================reward================")
        # print(neirewards)
        # print("--------------------------------------")
        # print(self.rewards[mytid])
        # print("--------------------------------------")

        for reward in neirewards.values():
            weighted_reward += neignbor_weight * reward

        # print(weighted_reward)
        # print("-------------diff---------------------")
        # print(self.rewards[mytid]-weighted_reward)
        # print("======================================")
        return weighted_reward


"""
该类用于包装agents
"""


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'name': "sumo_rl_v0",
        'is_parallelizable': True
    }

    def __init__(self, **kwargs):
        #        super(SumoEnvironmentPZ,self).__init__()
        EzPickle.__init__(self, **kwargs)  # 对象序列化与反序列化，将对象序列化为二进制文本。用于再造一个对象python pickle unpickle
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids

        self._agent_selector = agent_selector(self.agents)  # agent selector 是一个迭代器类，参数是agent list
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # for test===========#
        # self.pz_step = 0
        # self.resets = 0
        # self.acc_step = 0
        # ===================#

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}

        self.infos = {a: {} for a in self.agents}

    @property
    def unwrapped(self):
        return self.env

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # for test
        # self.resets = self.resets + 1
        # self.pz_step = 0 #每轮更新pettingzoo step

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        # print(self.env.observations)
        # print("========")
        # print(agent)
        # print("========")
        obs = self.env.observations[agent].copy()
        return obs

    def state(self):
        raise NotImplementedError('Method state() currently not implemented.')

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def save_csv(self, out_csv_name, run):
        self.env.save_csv(out_csv_name, run)

    def step(self, action):
        # self.pz_step = self.pz_step + 1
        # self.acc_step = self.acc_step + 1
        # print("pz step:",self.pz_step)
        # print("acc pz step:",self.acc_step)
        # print("pz resets:",self.resets)
        # print("env sim step:",self.env.sim_step)
        # print("dones:",self.dones)
        # print("sim max time:",self.env.sim_max_time)
        # sim step > sim max time之后，就会self.dones = {True}
        # 然后，self.was_done_step(action)就会逐渐删掉reward
        # 最后reward变成{}
        # ================================================#
        # 目前弄清楚的是：2022.9.25
        # sim step > max sim step 以后 dones={True}
        # 因为dones={True}，会有reset() → pz step = 0；sim step = 0
        # pz step和 sim step 大致应该是4:1的关系。但是，sim有一个time_to_load_vehicle = 10s，需要减掉。
        # 因此，有pz step = 36, sim step = 19(19-10=9,9*4=36)
        # 又如：pz step last agent: 352
        # sim step last agent: 98.0 98-10=88,88*4 = 352
        # 4是traffic signal的id数
        # ================================================#
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))
        self.env._apply_actions({agent: action})
        if self._agent_selector.is_last():  # 最后一个agent
            # 本句运行结束时time_to_act 必然是true
            # 执行步数恒等于delta_time
            self.env._run_steps()

            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()  # 此时time_to_act 必然是true，因此必有更新

            # ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # "两者必须一致"
            # print("rewards:",rewards)
            # print("env rewards:",self.env.rewards)
            # print("==================================")
            # ☆如果说self.env._run_steps()不能保证time_to_act 必然是true，则有以下结论
            # self.env._compute_observations()和self.env._compute_rewards()的返回值仅可用于测试，不可用于传递obs和reward值。
            # 测试指测试当前step是否update了数据。
            # (1)如果update了数据，即next_action_time = sim_step，则有len(obs)和len(reward)=agent number
            # (2)否则，len(obs)和len(reward)=0
            # self.rewards = self.env.rewards

            self.env._compute_lastmeasures()
            self.env._compute_info()
            self.env.last_reward = self.rewards.copy()  # copy极为重要，否则是传递指针，后面clear reward把reward给清零了 #此处应该是env.reward，否则reward可能有{}。此处取返回值似乎不妥。
        else:
            self._clear_rewards()

        done = self.env._compute_dones()['__all__']
        self.terminations = {a: done for a in self.agents}
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0  # 只需要存即时reward，所以清零也无妨
        self._accumulate_rewards()
