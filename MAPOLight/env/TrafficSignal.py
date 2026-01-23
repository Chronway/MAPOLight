# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 18:28:44 2023

@author: Wang Chong
"""
import random
from typing import Callable, List, Union
import traci
import numpy as np
#from gym import spaces ：too old version
from gymnasium import spaces


MAX_NEIGNBOR = 4 #
MAX_PHASE = 7

class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """
    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized? 
    MIN_GAP = 2.5
    
    def __init__(self, 
                env,
                ts_id: List[str],
                delta_time: int, 
                yellow_time: int, 
                min_green: int, 
                max_green: int,
                #phases,
                begin_time: int,
                reward_fn: Union[str,Callable],
                sumo):
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.phase_index = dict()
        
        if(env.cav_compare):
            #self.mean_diff = {'obs':None,'waits':None} #{'obs':,'waits':}
            #self.std_diff = {'obs':None,'waits':None} #{'obs':,'waits':}
            self.correlation = {'obs':None,'waits':None} #{'obs':,'waits':}
            self.diff_reward = {'result':None}
            self.last_measure_total = 0
        else:
            #print("NO CAV COMPARE!!!!!!!!!!!!!")
            pass

        self.phase_index = self.build_phases()
        
        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))

        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        if(self.env.collaborate):
            self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+3*len(self.lanes)+MAX_NEIGNBOR, dtype=np.float32), high=np.ones(self.num_green_phases+1+3*len(self.lanes)+MAX_NEIGNBOR, dtype=np.float32))
        else:
            self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+3*len(self.lanes), dtype=np.float32), high=np.ones(self.num_green_phases+1+3*len(self.lanes), dtype=np.float32))
        self.action_space = spaces.Discrete(self.num_green_phases) 
        
        self.netdata = self.env.netdata
        self.phase_lanes = self.phase_lanes(self.green_phases)
        self.max_pressure_lanes = self.max_pressure_lanes()

        self.sumo.junction.subscribeContext(self.id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                            [traci.constants.VAR_LANEPOSITION,
                                             traci.constants.VAR_SPEED,
                                             traci.constants.VAR_LANE_ID])
    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.phase_length = len(phases)
        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            yellow_state = ''
            for s in range(len(p1.state)):
                if (p1.state[s] == 'G'):
                    yellow_state += 'y'
                else:
                    yellow_state += p1.state[s]#'g'
            self.yellow_dict[i] = len(self.all_phases)
            self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases

        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)
        
        phases = dict()
        
        for index,phase in enumerate(self.all_phases):
            phases.update({phase.state:index})
        
        return phases

    def get_tl_green_phases(self):
        logic = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        green_phases = [p.state for p in logic.getPhases()
                        if 'y' not in p.state
                        and ('G' in p.state or 'g' in p.state)]

        return sorted(green_phases)

    @property
    def phase(self):
        return self.sumo.trafficlight.getPhase(self.id) #ok
    
    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def phase_lanes(self, actions):
        phase_lanes = {a.state: [] for a in actions}
        for a in actions:
            a = a.state
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])

            # some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    def max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g.state]:
                inc_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.add(ol)

            max_pressure_lanes[g.state] = {'inc': inc_lanes, 'out': out_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        # compute pressure for all green movements
        for g in self.green_phases:
            g = g.state
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            tl_data = self.sumo.junction.getContextSubscriptionResults(self.id)
            inc_pressure = sum([1 if tl_data[i][traci.constants.VAR_LANE_ID] == lane else 0 for i in tl_data for lane in inc_lanes])
            out_pressure = sum([1 if tl_data[i][traci.constants.VAR_LANE_ID] == lane else 0 for i in tl_data for lane in out_lanes ])
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)
        # print(phase_pressure, end='\t')
        # if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(self.green_phases):
            return self.get_phase_id(random.choice(self.green_phases).state)
        else:
            # choose phase with max pressure
            # if two phases have equivalent pressure
            # select one with more green movements
            # return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [(p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p: p[1], reverse=True)
            phase_pressure = [p for p in phase_pressure if p[1] == phase_pressure[0][1]]
            # print(self.get_phase_id(random.choice(phase_pressure)[0]), end='\t')
            # print(self.all_phases)
            return self.get_phase_id(random.choice(phase_pressure)[0])

    def get_phase_id(self, phase):
        """get phase id (int) from phase (String) like 'GGGrrrrrGGGrrrrr' """
        for i, p in enumerate(self.all_phases):
            if p.state == phase:
                return i
        assert False, f"can't find phase id ({phase})"

    def set_next_phase_random(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] or (String) like 'GGGrrrrrGGGrrrrr'
        """
        if type(new_phase) == str:
            new_phase = self.get_phase_id(new_phase)

        new_phase = int(new_phase)

        assert 0<=new_phase<self.num_green_phases, "selected phase out of range!"+str(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[self.green_phase]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_lastmeasure(self):
        return self.last_measure
    
    def compute_reward(self):
        if type(self.reward_fn) is str:
            if self.reward_fn == 'diff-waiting-time':
                self.last_reward = self._diff_waiting_time_reward()
                if(self.env.cav_compare):
                    self.last_total_reward = self._total_diff_waiting_time_reward()
                    self.diff_reward['result'] = self.compute_diffreward()
            elif self.reward_fn == 'average-speed':
                self.last_reward = self._average_speed_reward()
            elif self.reward_fn == 'queue':
                self.last_reward = self._queue_reward()
            elif self.reward_fn == 'pressure':
                self.last_reward = self._pressure_reward()
            else:
                raise NotImplementedError(f'Reward function {self.reward_fn} not implemented')
        else:
            self.last_reward = self.reward_fn(self)
        #print(self.id," DIFF:",self.diff_reward)
        return self.last_reward
    
    def compute_diffreward(self):
        if (abs(self.last_total_reward) + abs(self.last_reward))==0:
            return 0
        else:
            return 2*(abs(self.last_total_reward) - abs(self.last_reward))/(abs(self.last_total_reward) + abs(self.last_reward))

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
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward
    
    '''
    #get total waitting time for comparsion
    def get_waiting_time_per_lane(self):
        if(self.env.cav_env):
            cav_waittime = self.__get_waiting_time_per_lane_cav()
            if(self.env.cav_compare):
                all_waittime = self.__get_waiting_time_per_lane_all()
                self.mean_diff['waits'],self.std_diff['waits'] = self.__compare("wait",cav_waittime,all_waittime)
            return cav_waittime
        else:
            return self.__get_waiting_time_per_lane_all()
    '''
    
    def get_waiting_time_per_lane(self):
        if(self.env.cav_env):
            cav_waittime = self.__get_waiting_time_per_lane_cav()
            if(self.env.cav_compare):
                all_waittime = self.__get_waiting_time_per_lane_all()
                self.correlation['waits'] = self.__compare(cav_waittime,all_waittime)
            return cav_waittime
        else:
            return self.__get_waiting_time_per_lane_all()

    def __get_waiting_time_per_lane_all(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - \
                    sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return np.array(wait_time_per_lane)

    def __get_waiting_time_per_lane_cav(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list_all = self.sumo.lane.getLastStepVehicleIDs(lane)
            #==consider cav========#
            veh_list = list(self.env.cav_set.devids & set(veh_list_all))
            #print("veh list diff:",len(veh_list_all)-len(veh_list))
            #if(len(veh_list_all)>0):
                #print("cav percent per lane:",len(veh_list)/len(veh_list_all))
            #======================#
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return np.array(wait_time_per_lane)
    
    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return abs(sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap))
                for lane in self.out_lanes]

    def get_total_queued(self):
        return sum([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
    
    def get_energy_consumption_per_lane(self):
        energy_consumption_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            ecs = 0.0
            for veh in veh_list:
                ec = self.sumo.vehicle.getElectricityConsumption(veh)
                ecs += ec
            energy_consumption_per_lane.append(ecs)
        return energy_consumption_per_lane

    def _energy_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_energy = sum(self.traffic_signals[ts].get_energy_consumption())
            rewards[ts] = -ts_energy
            self.last_measure[ts] = ts_energy
        return rewards
    
    def get_avg_speed(self):
        spd_sum = 0
        count = 0
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                spd_sum += self.sumo.vehicle.getSpeed(veh)
                count += 1
        if(count == 0):
            return 0
        else:
            return spd_sum / count

    def compute_observation(self):
        if(self.env.cav_env):
            cav_obs = self.__compute_observation_cav()
            if(self.env.cav_compare):
                all_obs = self.__compute_observation_all()
                self.correlation['obs'] = self.__compare(cav_obs,all_obs)
            return cav_obs
        else:
            return self.__compute_observation_all()

    def __compute_observation_all(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        velocity = self.get_lanes_velocity()
        queue = self.get_lanes_queue()
        
        if(self.env.collaborate):
            neignbor_phases = self.get_neignbor_phases()
            observation = np.array(phase_id + min_green + density + velocity + queue + neignbor_phases, dtype=np.float32)
            #print("+++++++++",observation.shape)
            #return observation
        else:
            observation = np.array(phase_id + min_green + density + velocity + queue, dtype=np.float32)
        return observation
        #return observation
    
    def __compute_observation_cav(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        velocity,queue,density = self.get_dvlanes_info()
        if(self.env.collaborate):
            neignbor_phases = self.get_neignbor_phases()
            observation = np.array(phase_id + min_green + density + velocity + queue + neignbor_phases, dtype=np.float32)
        else:
            observation = np.array(phase_id + min_green + density + velocity + queue, dtype=np.float32)
        return observation

    def get_lanes_density(self):
        lanes_density = [self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in self.lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_velocity(self):
        lanes_veloctiy = [self.sumo.lane.getLastStepMeanSpeed(lane)/self.sumo.lane.getMaxSpeed(lane) for lane in self.lanes]
        return [min(1, veloctiy) for veloctiy in lanes_veloctiy]

    def get_lanes_queue(self):
        lanes_queue = [self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in self.lanes]
        return [min(1, queue) for queue in lanes_queue]
    
    def get_dvlanes_info(self):
        detected_vids = self.env.cav_set.devids

        dvlanes_velocity = list() #for all lanes of traffic signal
        dvlanes_queue = list() #for all lanes of traffic signal
        dvlanes_density = list() #for all lanes of traffic signal
        
        for lane in self.lanes:
            lane_vids = set(self.sumo.lane.getLastStepVehicleIDs(lane))
            lane_dvids = detected_vids & lane_vids
            dvlane_velocity = self.__get_dv_velocity(lane_dvids,lane)
            dvlane_halts = self.__get_dv_queue(lane_dvids,lane)
            dvlane_density = self.__get_dv_density(lane_dvids,lane)
            dvlanes_velocity.append(dvlane_velocity)
            dvlanes_queue.append(dvlane_halts)
            dvlanes_density.append(dvlane_density)
        return dvlanes_velocity,dvlanes_queue,dvlanes_density
    
    def __get_dv_density(self,devs:set,lane)->float:
        if(len(devs)==0): #no dev vehicle
            return 0
        vlens = list()
        for v in devs:
            l = self.sumo.vehicle.getLength(v)
            vlens.append(l)
        density = len(vlens) * (self.MIN_GAP + np.mean(vlens)) / self.lanes_length[lane]
        return min(1, density)

    def __get_dv_velocity(self,devs:set,lane)->float:
        if(len(devs)==0):
            return 1
        vvs = list() #vehicle velocity
        for v in devs:
            vv= self.sumo.vehicle.getSpeed(v)
            vvs.append(vv)
        veloctiy = np.mean(vvs)/self.sumo.lane.getMaxSpeed(lane)
        return min(1, veloctiy)

    def __get_dv_queue(self,devs:set,lane)->float:
        if(len(devs)==0):
            return 0
        halts = 0 # halted vehicles
        vlens = list() #vehicle length
        for v in devs:
            vv= self.sumo.vehicle.getSpeed(v)
            l = self.sumo.vehicle.getLength(v)
            vlens.append(l)
            if(vv<=0.1):
                halts += 1
        queue = halts * (self.MIN_GAP + np.mean(vlens)) / self.lanes_length[lane]
        return min(1, queue)
    
    def __compare(self,cav:np.ndarray,allv:np.ndarray):
        correlation = np.corrcoef(allv, cav)  # > 0.8
        return correlation
    
    
    def get_neignbor_phases(self):
        if self.env.collaborate:
            neignbor_tids = self.env.neignbors[self.id]
            neignbor_phases = np.zeros((MAX_NEIGNBOR,))
            for i,neignbor in enumerate(neignbor_tids):
                #print(neignbor)
                max_index = len(self.all_phases)-1
                assert max_index>0, "max index less or equal to zero!!"
                neignbor_phases[i] = self.phase_index[self.sumo.trafficlight.getRedYellowGreenState(neignbor)]/max_index
                #print("max phase index:",(len(self.all_phases)-1))
            return neignbor_phases.tolist()
        else:
            return list()