# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 18:28:44 2023

@author: Wang Chong
"""
import random
from typing import Callable, List, Literal, Optional, Union
import traci
import numpy as np

# from gym import spaces ：too old version
from gymnasium import spaces

from typing import TYPE_CHECKING

import traci.constants

from env.networkdata import NetworkData

if TYPE_CHECKING:
    from env.SignalEnv import SumoEnvironmentPZ

MAX_NEIGNBOR = 4  #
MAX_PHASE = 7


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
            self,
            env,
            ts_id: str,
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            begin_time: int,
            reward_fn: Union[str, Callable],
            collaborate: bool,
            netdata: dict,
            cav_env: bool,
            cav_compare: bool,
    ):
        self.id = ts_id
        self.env: SumoEnvironmentPZ = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green  # kl1nge5: 根本没用到
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.reward_fn = reward_fn
        self.collaborate: bool = collaborate
        self.cav_env: bool = cav_env
        self.cav_compare: bool = cav_compare
        self.phase_index = dict()

        if env.cav_compare:
            # self.mean_diff = {'obs':None,'waits':None} #{'obs':,'waits':}
            # self.std_diff = {'obs':None,'waits':None} #{'obs':,'waits':}
            self.correlation: dict[Literal["obs", "waits"], Optional[np.ndarray]] = {"obs": None,
                                                                                     "waits": None}  # {'obs':,'waits':}
            self.diff_reward: dict[Literal["result"], Optional[float]] = {"result": None}  # 设为字典，否则不能传递到signalenv里
            self.last_measure_total = 0
        else:
            # print("NO CAV COMPARE!!!!!!!!!!!!!")
            pass

        self.phase_index = self.build_phases()
        self.in_lanes = list(
            dict.fromkeys(traci.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = list(set([
            link[0][1]
            for link in traci.trafficlight.getControlledLinks(self.id)
            if link
        ]))

        self.lanes_length = {
            lane: traci.lane.getLength(lane) for lane in self.in_lanes + self.out_lanes
        }

        if self.collaborate:
            self.observation_space = spaces.Box(
                low=np.zeros(
                    self.num_green_phases + 1 + 3 * len(self.in_lanes) + MAX_NEIGNBOR,
                    dtype=np.float32,
                ),
                high=np.ones(
                    self.num_green_phases + 1 + 3 * len(self.in_lanes) + MAX_NEIGNBOR,
                    dtype=np.float32,
                ),
            )
            # self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+3*len(self.lanes)+8, dtype=np.float32), high=np.ones(self.num_green_phases+1+3*len(self.lanes)+8, dtype=np.float32))
        else:
            self.observation_space = spaces.Box(
                low=np.zeros(
                    self.num_green_phases + 1 + 3 * len(self.in_lanes), dtype=np.float32
                ),
                high=np.ones(
                    self.num_green_phases + 1 + 3 * len(self.in_lanes), dtype=np.float32
                ),
            )
        self.action_space = spaces.Discrete(self.num_green_phases)

        """
        build phase:
        green phase
        [Phase(duration=60, state='GGGgrrrrGGGgrrrr', minDur=-1, maxDur=-1, next=()),
         Phase(duration=60, state='rrrGrrrrrrrGrrrr', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=60, state='rrrrGGGgrrrrGGGg', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=60, state='rrrrrrrGrrrrrrrG', minDur=-1, maxDur=-1, next=())]
        
        all phase
        [Phase(duration=60, state='GGGgrrrrGGGgrrrr', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=60, state='rrrGrrrrrrrGrrrr', minDur=-1, maxDur=-1, next=()),
         Phase(duration=60, state='rrrrGGGgrrrrGGGg', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=60, state='rrrrrrrGrrrrrrrG', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=3, state='yyygrrrryyygrrrr', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=3, state='rrryrrrrrrryrrrr', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=3, state='rrrryyygrrrryyyg', minDur=-1, maxDur=-1, next=()), 
         Phase(duration=3, state='rrrrrrryrrrrrrry', minDur=-1, maxDur=-1, next=())]
        self.yellow_dict → {0: 4, 1: 5, 2: 6, 3: 7}
        """

        self.netdata = netdata
        self.phase_lanes = self.build_phase_lanes(self.green_phases)
        self.max_pressure_lanes = self.build_max_pressure_lanes()
        # print(self.max_pressure_lanes)
        traci.junction.subscribeContext(
            self.id,
            traci.constants.CMD_GET_VEHICLE_VARIABLE,
            150,
            [
                traci.constants.VAR_LANEPOSITION,
                traci.constants.VAR_SPEED,
                traci.constants.VAR_LANE_ID,
            ],
        )

    def build_phases(self):
        phases = traci.trafficlight.getAllProgramLogics(self.id)[0].phases
        # if self.env.fixed_ts:
        #   self.num_green_phases = len(phases)//2  # Number of green phases == number of phases (green+yellow) divided by 2
        #   return
        self.phase_length = len(phases)
        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (
                    state.count("r") + state.count("s") != len(state)
            ):  # r+s=state说明是全红相位
                self.green_phases.append(traci.trafficlight.Phase(60, state))  # type: ignore
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            yellow_state = ""
            for s in range(len(p1.state)):
                if p1.state[s] == "G":
                    yellow_state += "y"
                else:
                    yellow_state += p1.state[s]  # 'g'
            self.yellow_dict[i] = len(self.all_phases)
            self.all_phases.append(
                traci.trafficlight.Phase(self.yellow_time, yellow_state)  # type: ignore
            )

        # print(self.all_phases)
        # print("============")
        # print(self.green_phases)
        # print("============")
        # print(self.num_green_phases)
        # print("============")
        # print(self.yellow_dict)

        programs = traci.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases

        # print("########################################")
        # print(self.id)
        # print(logic.phases)
        # print("########################################")
        traci.trafficlight.setProgramLogic(self.id, logic)
        traci.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

        # print("==============")
        phases = dict()

        for index, phase in enumerate(self.all_phases):
            phases.update({phase.state: index})

        return phases
        # raise NameError

    def get_tl_green_phases(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        # get only the green phases
        green_phases = [
            p.state
            for p in logic.getPhases()
            if "y" not in p.state and ("G" in p.state or "g" in p.state)
        ]

        # sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)  # ok

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.current_simulation_time

    def update(self):
        """
        每过一秒需要调用一次该方法，用以实现黄灯计时，并在黄灯结束后将信号灯设置为下一个相位
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.green_phase].state  # 严谨来说，all_phases应改为green_phases更符合语义。但是这样写也没错，因为在结构上green_phases刚好是all_phases的前缀。
            )
            self.is_yellow = False

    def build_phase_lanes(self, actions):
        phase_lanes = {a.state: [] for a in actions}
        for a in actions:
            a = a.state
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                # print(self.netdata["inter"][self.id]["tlsindex"])
                if a[s] == "g" or a[s] == "G":
                    green_lanes.add(self.netdata["inter"][self.id]["tlsindex"][s])
                elif a[s] == "r":
                    red_lanes.add(self.netdata["inter"][self.id]["tlsindex"][s])

            # some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    def build_max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            g = g.state
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                phase_green_idx = _find_all_indices(g.upper(), "G")
                inc_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']:
                    if self.netdata['lane'][l]['outgoing'][ol]['index'] in phase_green_idx:
                        out_lanes.add(ol)

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        #compute pressure for all green movements
        for g in self.green_phases:
            g = g.state
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            #pressure is defined as the number of vehicles in a lane
            inc_pressure = sum([ traci.lane.getLastStepVehicleNumber(l) for l in inc_lanes])
            out_pressure = sum([ traci.lane.getLastStepVehicleNumber(l) for l in out_lanes])
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)

        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(self.green_phases):
            return random.choice(range(len(self.green_phases)))
        else:
            #choose phase with max pressure
            #if two phases have equivalent pressure
            #select one with more green movements
            #return max(phase_pressure, key=lambda p:phase_pressure[p])
            keys = list(phase_pressure.keys())
            weights = list(phase_pressure.values())
            weights = [w if w > 0 else 0.0001 for w in weights]
            # 使用 random.choices 进行加权采样（例如采样一个）
            # print(weights)
            phase_pressure = random.choices(keys, weights=weights, k=1)[0]
            for i, e in enumerate(self.green_phases):
                if e.state == phase_pressure:
                    return i

    def get_phase_id(self, phase):
        """get phase id (int) from phase (String) like 'GGGrrrrrGGGrrrrr'"""
        for i, p in enumerate(self.all_phases):
            if p.state == phase:
                return i
        assert False, f"can't find phase id ({phase})"

    def set_next_phase(self, new_phase: np.int64):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (np.int64): Number between [0 ... num_green_phases]
        Kl1nge5: 该方法只在环境step中被调用，由于限制delta_time>yellow_time，所以每次调用该方法时信号灯应该必然不是黄灯
        故此时信号灯的相位一定是self.green_phase
        """
        new_phase = int(new_phase)  # type: ignore
        assert not self.is_yellow
        if (  # 如果当前相位不等于新相位，且已经度过了最小绿灯时间
                self.green_phase != new_phase
                and self.time_since_last_phase_change > self.yellow_time + self.min_green
        ):  # 就转变为黄灯，准备切换新相位
            traci.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[self.green_phase]].state
            )
            self.green_phase = new_phase
            self.is_yellow = True
            self.time_since_last_phase_change = 0
        self.next_action_time = self.env.current_simulation_time + self.delta_time

    def compute_reward(self):
        if type(self.reward_fn) is str:
            if self.reward_fn == "diff-waiting-time":
                self.last_reward = self._diff_waiting_time_reward()
                if self.cav_compare:
                    self.last_total_reward = self._total_diff_waiting_time_reward()
                    self.diff_reward["result"] = (
                        self.compute_diffreward()
                    )  # 只是计算all和cav环境下的reward差值
            elif self.reward_fn == "average-speed":
                self.last_reward = self._average_speed_reward()
            elif self.reward_fn == "queue":
                self.last_reward = self._queue_reward()
            elif self.reward_fn == "pressure":
                self.last_reward = self._pressure_reward()
            else:
                raise NotImplementedError(
                    f"Reward function {self.reward_fn} not implemented"
                )
        else:
            assert callable(self.reward_fn)
            self.last_reward = self.reward_fn(self)
        # print(self.id," DIFF:",self.diff_reward)
        return self.last_reward

    def compute_diffreward(self):
        if (abs(self.last_total_reward) + abs(self.last_reward)) == 0:
            return 0.
        else:
            return (
                    2
                    * (abs(self.last_total_reward) - abs(self.last_reward))
                    / (abs(self.last_total_reward) + abs(self.last_reward))
            )

    def _pressure_reward(self):
        return -self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _total_diff_waiting_time_reward(self):
        ts_wait = sum(self.__get_waiting_time_per_lane_all()) / 100.0
        reward = self.last_measure_total - ts_wait
        self.last_measure_total = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0 / ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        if self.cav_env:
            cav_waittime = self.__get_waiting_time_per_lane_cav()
            if self.cav_compare:
                all_waittime = self.__get_waiting_time_per_lane_all()
                # self.correlation['waits'] = self.__compare("wait",cav_waittime,all_waittime)
                self.correlation["waits"] = self.__compare(cav_waittime, all_waittime)
            return cav_waittime
        else:
            return self.__get_waiting_time_per_lane_all()

    def __get_waiting_time_per_lane_all(self):
        wait_time_per_lane = []
        for lane in self.in_lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.all_vehicles_info:
                    self.env.all_vehicles_info[veh] = {veh_lane: acc}
                else:
                    self.env.all_vehicles_info[veh][veh_lane] = acc - sum(
                        [
                            self.env.all_vehicles_info[veh][lane]
                            for lane in self.env.all_vehicles_info[veh].keys()
                            if lane != veh_lane
                        ]
                    )
                wait_time += self.env.all_vehicles_info[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return np.array(wait_time_per_lane)

    def __get_waiting_time_per_lane_cav(self):
        assert self.env.cav_set is not None  # for type hint
        wait_time_per_lane = []
        for lane in self.in_lanes:
            veh_list_all = traci.lane.getLastStepVehicleIDs(lane)
            # ==consider cav========#
            veh_list = list(self.env.cav_set.devids & set(veh_list_all))
            # print("veh list diff:",len(veh_list_all)-len(veh_list))
            # if(len(veh_list_all)>0):
            # print("cav percent per lane:",len(veh_list)/len(veh_list_all))
            # ======================#
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.all_vehicles_info:
                    self.env.all_vehicles_info[veh] = {veh_lane: acc}
                else:
                    self.env.all_vehicles_info[veh][veh_lane] = acc - sum(
                        [
                            self.env.all_vehicles_info[veh][lane]
                            for lane in self.env.all_vehicles_info[veh].keys()
                            if lane != veh_lane
                        ]
                    )
                wait_time += self.env.all_vehicles_info[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return np.array(wait_time_per_lane)

    # ==========================================#保留#============================================#
    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += traci.vehicle.getSpeed(v) / traci.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return abs(
            sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.in_lanes)
            - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        )

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [
            min(
                1,
                traci.lane.getLastStepVehicleNumber(lane)
                / (self.lanes_length[lane] / vehicle_size_min_gap),
            )
            for lane in self.out_lanes
        ]

    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes])

    def get_respective_queued(self):
        return {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in self.in_lanes}


    def _get_veh_list(self):
        veh_list = []
        for lane in self.in_lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    # ===================自己添加===========================================#
    def get_energy_consumption_per_lane(self):
        energy_consumption_per_lane = []
        for lane in self.in_lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            ecs = 0.0
            for veh in veh_list:
                ec = traci.vehicle.getElectricityConsumption(veh)
                ecs += ec
            energy_consumption_per_lane.append(ecs)
        return energy_consumption_per_lane

    def get_avg_speed(self):
        spd_sum = 0
        count = 0
        for lane in self.in_lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                spd_sum += traci.vehicle.getSpeed(veh)
                count += 1
        if count == 0:
            return 50
        else:
            return spd_sum / count

    # =========================================================================================#
    def compute_observation(self):
        if self.cav_env:
            cav_obs = self.__compute_observation_cav()
            if self.cav_compare:
                all_obs = self.__compute_observation_all()
                self.correlation["obs"] = self.__compare(cav_obs, all_obs)
            return cav_obs
        else:
            return self.__compute_observation_all()

    def __compute_observation_all(self):
        phase_id = [
            1 if self.green_phase == i else 0 for i in range(self.num_green_phases)
        ]  # one-hot encoding
        min_green = [
            (
                0
                if self.time_since_last_phase_change < self.min_green + self.yellow_time
                else 1
            )
        ]
        density = self.get_lanes_density()
        velocity = self.get_lanes_velocity()
        queue = self.get_lanes_queue()

        if self.collaborate:
            neignbor_phases = self.get_neighbor_phases()
            observation = np.array(
                phase_id + min_green + density + velocity + queue + neignbor_phases,
                dtype=np.float32,
            )
            # print("+++++++++",observation.shape)
            # return observation
        else:
            observation = np.array(
                phase_id + min_green + density + velocity + queue, dtype=np.float32
            )
        return observation
        # return observation

    def __compute_observation_cav(self):
        phase_id = [
            1 if self.green_phase == i else 0 for i in range(self.num_green_phases)
        ]  # one-hot encoding
        min_green = [
            (
                0
                if self.time_since_last_phase_change < self.min_green + self.yellow_time
                else 1
            )
        ]
        velocity, queue, density = self.get_dvlanes_info()
        # observation = np.array(phase_id + min_green + density + velocity + queue, dtype=np.float32)

        # if(np.shape(observation)==31):
        #    print("@@@@@@@@@@@@@@@@")
        #    observation = observation+np.zeros(10)
        # print("obs shape:",np.shape(observation))
        if self.collaborate:
            neignbor_phases = self.get_neighbor_phases()
            # print(neignbor_phases)
            # print(type(neignbor_phases))
            # print(type(phase_id))
            observation = np.array(
                phase_id + min_green + density + velocity + queue + neignbor_phases,
                dtype=np.float32,
            )
            # print(observation)
            # print("+++++++++",observation.shape)
            # return observation
        else:
            observation = np.array(
                phase_id + min_green + density + velocity + queue, dtype=np.float32
            )
        return observation

    def get_lanes_density(self):
        lanes_density = [
            traci.lane.getLastStepVehicleNumber(lane)
            / (
                    self.lanes_length[lane]
                    / (self.MIN_GAP + traci.lane.getLastStepLength(lane))
            )
            for lane in self.in_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_velocity(self):
        lanes_veloctiy = [
            traci.lane.getLastStepMeanSpeed(lane) / traci.lane.getMaxSpeed(lane)
            for lane in self.in_lanes
        ]
        return [min(1, veloctiy) for veloctiy in lanes_veloctiy]

    def get_lanes_queue(self):
        lanes_queue = [
            traci.lane.getLastStepHaltingNumber(lane)
            / (
                    self.lanes_length[lane]
                    / (self.MIN_GAP + traci.lane.getLastStepLength(lane))
            )
            for lane in self.in_lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_dvlanes_info(self):
        assert self.env.cav_set is not None
        detected_vids = self.env.cav_set.devids

        dvlanes_velocity = list()  # for all lanes of traffic signal
        dvlanes_queue = list()  # for all lanes of traffic signal
        dvlanes_density = list()  # for all lanes of traffic signal

        for lane in self.in_lanes:
            lane_vids = set(traci.lane.getLastStepVehicleIDs(lane))
            lane_dvids = detected_vids & lane_vids
            dvlane_velocity = self.__get_dv_velocity(lane_dvids, lane)
            dvlane_halts = self.__get_dv_queue(lane_dvids, lane)
            dvlane_density = self.__get_dv_density(lane_dvids, lane)
            dvlanes_velocity.append(dvlane_velocity)
            dvlanes_queue.append(dvlane_halts)
            dvlanes_density.append(dvlane_density)
        return dvlanes_velocity, dvlanes_queue, dvlanes_density

    def __get_dv_density(self, devs: set, lane) -> float:
        if len(devs) == 0:  # no dev vehicle
            return 0
        vlens = list()
        for v in devs:
            l = traci.vehicle.getLength(v)
            vlens.append(l)
        density = len(vlens) * (self.MIN_GAP + np.mean(vlens)) / self.lanes_length[lane]
        return min(1, density)

    def __get_dv_velocity(self, devs: set, lane) -> float:
        if len(devs) == 0:
            return 1
        vvs = list()  # vehicle velocity
        for v in devs:
            vv = traci.vehicle.getSpeed(v)
            vvs.append(vv)
        # Kl1nge5: 这里存在一个未定义的行为：当道路上没有智能车时，vvs为空，均值结果为nan，min(1, velocity)返回1
        velocity = np.mean(vvs) / traci.lane.getMaxSpeed(lane)
        return min(1, velocity)

    def __get_dv_queue(self, devs: set, lane) -> float:
        if len(devs) == 0:
            return 0
        halts = 0  # halted vehicles
        vlens = list()  # vehicle length
        for v in devs:
            vv = traci.vehicle.getSpeed(v)
            l = traci.vehicle.getLength(v)
            vlens.append(l)
            if vv <= 0.1:
                halts += 1
        queue = halts * (self.MIN_GAP + np.mean(vlens)) / self.lanes_length[lane]
        return min(1, queue)

    def __compare(self, cav: np.ndarray, allv: np.ndarray):
        correlation = np.corrcoef(allv, cav)  # > 0.8
        # TODO line

        # print(np.nanmean(correlation))
        return correlation

    def get_neighbor_phases(self):
        if self.collaborate:
            neighbor_ids = self.env.neighbors[self.id]
            neighbor_phases = np.zeros((MAX_NEIGNBOR,))
            for i, neighbor in enumerate(neighbor_ids):
                max_index = len(self.all_phases) - 1
                assert max_index > 0, "max index less or equal to zero!!"
                neighbor_phases[i] = (
                        self.phase_index[
                            traci.trafficlight.getRedYellowGreenState(neighbor)
                        ]
                        / max_index
                )
                # print("max phase index:",(len(self.all_phases)-1))
            return neighbor_phases.tolist()  # 需要是list
        else:
            return list()


def _find_all_indices(s, char):
    return [i for i, c in enumerate(s) if c == char]