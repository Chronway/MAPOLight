# -*- coding: utf-8 -*-
import functools
import os, sys
from typing import Callable, Union, Optional
from gymnasium.utils import EzPickle
from pettingzoo.utils.env import AgentID

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import pandas as pd
import sumolib
import typing
import traci

from utils.MobileSensor import CAVs

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .TrafficSignal import TrafficSignal
from .networkdata import NetworkData


def env(**kwargs):
    _env = SumoEnvironmentPZ(**kwargs)
    _env = wrappers.AssertOutOfBoundsWrapper(_env)
    _env = wrappers.OrderEnforcingWrapper(_env)
    return _env


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {
        "name": "sumo_v1"
    }

    def __init__(
            self,
            net_file: str,  # 路网文件
            route_file: str,  # 路由文件
            pr: float = 0.05,  # 渗透率
            use_gui: bool = False,  # 是否显示gui
            begin_time: int = 0,  # 智能体开始介入的时间
            num_seconds: int = 20000,  # 模拟的总时间
            max_depart_delay: int = 100000,  # sumo参数，见sumo文档
            time_to_teleport: int = -1,  # sumo参数，见sumo文档
            delta_time: int = 5,  # 智能体决策的间隔时间
            yellow_time: int = 3,  # 黄灯的持续时间
            min_green: int = 5,  # 最小绿灯持续时间，如果当前动作选择改变相位，但当前的绿灯未持续足够时间，则当前动作失效
            max_green: int = 50,  # sumo-rl遗留代码，根本没用到
            reward_fn: Union[str, Callable] = "diff-waiting-time",  # 奖励函数
            sumo_warnings: bool = True,  # 是否显示sumo警告
            cav_env: bool = False,  # 是否启动cav
            # be careful：cav compare will consume alot of resources and make the training intolerable long!!!
            cav_compare: bool = False,  # 是否启动cav比较
            collaborate: bool = False,  # 是否启动智能体协作：使用协作观察和协作奖励
            fcd_path: Optional[str] = None,  # 不为None时，sumo会输出 {fcd_path}/fcd_{run}.xml 文件，该文件可以用于绘制td图
            # 评估时使用——添加事故，如果不要添加事故，则保持下列参数为None
            # 在经过accident_time秒后，会尝试在accident_edge边选择一辆车停车以模拟一场事故
            # 未必精确在accident_time时发生事故，因为当时可能该路段没车，或是没有停车的条件，若如此则会每秒进行一次尝试直至成功制造一次事故
            accident_edge: Optional[list] = None,  # 事故发生的路段
            accident_time: Optional[list] = None,  # 事故发生的时间
            # 训练时使用
            accident_num: Optional[int] = None,  # 随机发生事故的次数
            # 通用
            accident_duration: Optional[int] = None,  # 事故持续时间
            scale: str = "1",  # 流量倍率，如果>1，则每次插入车时，会多插入车，如果<1，则每次插入车时，有一定概率不插入
    ):
        EzPickle.__init__(self, net_file, route_file, pr, use_gui, begin_time, num_seconds, max_depart_delay,
                          time_to_teleport,
                          delta_time, yellow_time, min_green, max_green, reward_fn, sumo_warnings, cav_env, cav_compare,
                          collaborate,
                          fcd_path, accident_edge, accident_time, accident_num, accident_duration)

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.sumo_binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.sumo_seed = None  # sumo_seed 于 reset 时设置
        self.sumo_warnings = sumo_warnings
        self.fcd_path = fcd_path
        self.scale = scale
        assert accident_num is None or (accident_edge is None and accident_time is None)  # 随机生成事故和固定生成事故只能选一个
        self.accident_num = accident_num
        self.accident_edge: Optional[list] = accident_edge
        self.accident_time: Optional[list] = accident_time
        # 上面的属性是用以保存参数的，下面的属性是每次模拟中使用的，会在模拟中被修改，所以reset时需要重置
        self.left_accident_edge: Optional[list]
        self.left_accident_time: Optional[list]
        self.accident_position: list  # 记录每次事故发生的位置
        assert self.accident_time is None or accident_duration is not None  # 如果设置了accident_time就必须设置accident_duration
        self.accident_duration = accident_duration

        assert delta_time > yellow_time  # 避免在黄灯时改相位
        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        assert cav_env or pr == 1  # 禁止没有开启cav_env却设置了渗透率(渗透率!=1)
        assert cav_env or not cav_compare  # 禁止没有开启cav_env却开启cav_compare
        self.cav_env = cav_env
        self.cav_compare = cav_compare
        if self.cav_env:
            print("########################################")
            print("####### CAV environment started!!! #####")
            self.cav_set = CAVs(pr, net_file)
        else:
            print("########################################")
            print("#######    NO CAV environment!!!   #####")
            self.cav_set = None
        if self.cav_compare:
            print("#######   CAV compare started!!!   #####")
            print("########################################")
        else:
            print("#######    NO CAV compare!!!       #####")
            print("########################################")
        self.reward_fn = reward_fn
        self.collaborate = collaborate

        self.netdata = NetworkData(self.net_file).get_net_data()  # 构造traffic_signal会用到

        self._start_dummy_simulation()
        self.possible_agents = list(traci.trafficlight.getIDList())
        self.agents = self.possible_agents
        self.traffic_signals = {
            tid: TrafficSignal(
                env=self,
                ts_id=tid,
                delta_time=self.delta_time,
                yellow_time=self.yellow_time,
                min_green=self.min_green,
                max_green=self.max_green,
                begin_time=self.begin_time,
                reward_fn=self.reward_fn,
                collaborate=self.collaborate,
                netdata=self.netdata,
                cav_env=self.cav_env,
                cav_compare=self.cav_compare,
            )
            for tid in self.agents
        }
        self._end_dummy_simulation()

        self.all_vehicles_info = dict()  # 会被所有TrafficSignal共同维护，用以记录车辆的信息

        self.run = -1  # 运行次数，reset后+1
        self.observations: dict
        self.last_action: dict
        self.last_diff = None

        if self.collaborate:
            self.neighbors: dict = {}
            self._load_neighbors()

        self.observation_spaces: dict = {tid: self.traffic_signals[tid].observation_space for tid in self.agents}
        self.action_spaces: dict = {tid: self.traffic_signals[tid].action_space for tid in self.agents}
        self.terminations: dict
        self.truncations: dict
        self.rewards: dict
        self._cumulative_rewards: dict
        self.infos: dict
        self.summary: list = []  # 辅助变量，储存每步的信息，包含指标、智能体动作、观察等，最后通过save_csv保存成文件
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection: str

        self.running_simulation = False  # 标注是否已经_start_simulation


    def _start_dummy_simulation(self):
        """
        用于获取信号灯信息的假模拟
        """
        sumo = sumolib.checkBinary("sumo")
        cmd = [sumo, "-n", self.net_file, "-r", self.route_file, ]
        traci.start(cmd)

    def _end_dummy_simulation(self):
        traci.close()

    def _start_simulation(self):
        sumo_cmd = [
            self.sumo_binary,
            "-n",
            self.net_file,
            "-r",
            self.route_file,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",  # 该参数决定了API::getAccumulatedWaitingTime的统计窗口大小
            str(self.sim_max_time),
            "--time-to-teleport",
            str(self.time_to_teleport),
            "--scale",
            str(self.scale)
        ]
        if self.sumo_seed is None:
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        # if self.use_gui:
        #     sumo_cmd.extend(["--start", "--quit-on-end"])
        if self.fcd_path is not None:
            os.makedirs(self.fcd_path, exist_ok=True)
            sumo_cmd.extend(["--fcd-output", f"{self.fcd_path}/fcd_{self.run}.xml"])
        traci.start(sumo_cmd)
        self.running_simulation = True

    def reset(self, seed=None, options=None, ):
        # 如果在reset指定了当前run的序号，则使用指定的序号，否则将上一次的序号+1，序号是用于输出文件名的
        if options is not None and "run" in options:
            self.run = options["run"]
        else:
            self.run += 1

        # 重置模拟
        if self.running_simulation:
            traci.close()
        self.sumo_seed = seed
        if seed is not None:
            np.random.seed(int(seed))
        self._start_simulation()
        # 重置信号灯对象
        self.traffic_signals = {
            tid: TrafficSignal(
                env=self,
                ts_id=tid,
                delta_time=self.delta_time,
                yellow_time=self.yellow_time,
                min_green=self.min_green,
                max_green=self.max_green,
                begin_time=self.begin_time,
                reward_fn=self.reward_fn,
                collaborate=self.collaborate,
                netdata=self.netdata,
                cav_env=self.cav_env,
                cav_compare=self.cav_compare,
            )
            for tid in self.agents
        }
        # 重置环境状态
        self.agent_selection = self._agent_selector.reset()
        self.terminations: dict = {tid: False for tid in self.agents}
        self.truncations: dict = {tid: False for tid in self.agents}
        self.rewards: dict = {tid: 0. for tid in self.agents}
        self._cumulative_rewards: dict = {tid: 0. for tid in self.agents}
        self.infos: dict = {tid: {} for tid in self.agents}
        self.summary.clear()
        # 重置事故变量
        if self.accident_num is not None:  # 如果事故时间是随机生成的
            self.left_accident_time = sorted(  # 则随机生成事故时间
                list(np.random.choice(list(range(self.begin_time, self.sim_max_time - self.accident_duration)), self.accident_num)))
            print("Generated accident time:", self.left_accident_time)
        else:  # 否则使用预先定义的事故时间
            self.left_accident_time = self.accident_time.copy() if self.accident_time is not None else None
        self.left_accident_edge = self.accident_edge.copy() if self.accident_edge is not None else None
        self.accident_position = []
        # 重置智能体观察和上次选择的动作
        self.observations: dict = {ts: None for ts in self.agents}
        self.last_action: dict = {ts: None for ts in self.agents}
        # 重置某些记录
        if self.cav_compare:
            self.last_diff = {
                ts: {
                    "correlation": self.traffic_signals[ts].correlation,
                    "diffreward": self.traffic_signals[ts].diff_reward,
                }
                for ts in self.agents
            }
        # 重置被traffic_signal改变过东西
        self.all_vehicles_info.clear()
        # Load vehicles
        for _ in range(self.begin_time):
            self._sumo_step()
        # 首次观察
        for tid in self.agents:
            self.observations[tid] = self.traffic_signals[tid].compute_observation()

    @property
    def current_simulation_time(self):
        """
        Return current simulation second on SUMO
        """
        return typing.cast(int, traci.simulation.getTime())  # typing.cast是用于标注类型的，便于编辑器代码提示，没有实际作用

    def step(self, action: np.int64):
        agent = self.agent_selection
        assert self.traffic_signals[
            agent].time_to_act, f"{self.traffic_signals[agent].next_action_time} ?= {self.current_simulation_time}"
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        self.traffic_signals[agent].set_next_phase(action)
        self.last_action[agent] = action

        if self._agent_selector.is_last():
            while not self.traffic_signals[agent].time_to_act:  # 因为所有信号灯都是同时行动的，这里用哪个都一样
                self._sumo_step()
                for ts in self.agents:
                    self.traffic_signals[ts].update()

            # update observation & reward
            if self.cav_env:
                self.cav_set.update_detects()  # type: ignore
            for tid in self.agents:
                assert self.traffic_signals[tid].time_to_act
                self.observations[tid] = self.traffic_signals[tid].compute_observation()
                self.rewards[tid] = self.traffic_signals[tid].compute_reward()

            if self.collaborate:
                new_reward = dict()
                for tid in self.agents:
                    new_reward[tid] = self.mixed_reward(tid)
                self.rewards.update(new_reward)

            # check termination
            if self.current_simulation_time > self.sim_max_time:
                for tid in self.agents:
                    self.terminations[tid] = True

            self.summary.append(self._compute_step_info())

        self._accumulate_rewards()  # 将 rewards 累积到 cumulate_rewards 中
        self.agent_selection = self._agent_selector.next()

    def max_pressure_action(self, tid):
        return self.traffic_signals[tid].max_pressure()

    @functools.lru_cache(maxsize=None)  # 缓存函数结果，提高性能
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.observations[agent]

    def _sumo_step(self):
        if self.left_accident_time is not None and len(self.left_accident_time) != 0:  # 需要发生事故，无论是固定还是随机
            if self.current_simulation_time > int(self.left_accident_time[0]):  # 需要发生事故
                if self.left_accident_edge is None:
                    candidate_vehicles = [i for i in traci.vehicle.getIDList()]
                else:
                    candidate_vehicles = [i for i in traci.vehicle.getIDList() if
                                          traci.vehicle.getRoadID(i) == self.left_accident_edge[0]]
                if len(candidate_vehicles) > 0:
                    accident_vid = np.random.choice(candidate_vehicles, 1).item()  # 随机选取一辆车
                    accident_edge = traci.vehicle.getRoadID(accident_vid)
                    accident_position = traci.vehicle.getLanePosition(accident_vid) + 50  # 当前车位置的50米后  # type: ignore
                    try:
                        traci.vehicle.setStop(
                            vehID=accident_vid,
                            edgeID=accident_edge,
                            pos=accident_position,
                            laneIndex=traci.vehicle.getLaneIndex(accident_vid), # type: ignore
                            duration=self.accident_duration, # type: ignore
                        )
                        traci.vehicle.setColor(accident_vid, (255, 0, 0, 255))
                        self.accident_position.append(accident_position)
                        print(f"Accident created successfully in {accident_edge}:{accident_position}")
                        if self.left_accident_edge is not None:  # 如果是随机生成，left_accident_edge为None
                            self.left_accident_edge.pop(0)
                        self.left_accident_time.pop(0)
                    except Exception as e:
                        print("WARNING: create accident failed: ", e)
                else:
                    print("WARNING: No vehicles in target edge to make accident")
        traci.simulationStep()
        if self.cav_env:  # 定义了车联网环境再跑
            self.cav_set.update_observations()  # 更新三集合：车集合，cav车集合，cav探测到的车集合  # type: ignore

    def _compute_step_info(self):
        all_vehs_in_env = traci.vehicle.getIDList()
        if len(all_vehs_in_env) == 0:
            return {
                "t": self.current_simulation_time,
                "wt": 0,
                "ql": 0,
                "sd": -1,
                # "b3c3_sd": "-1",
                # "b3c3_ql": "0",
            }
        else:
            mean_waitting_time = np.mean([traci.vehicle.getAccumulatedWaitingTime(vid) for vid in all_vehs_in_env])
            mean_queue_length = np.mean([self.traffic_signals[tid].get_total_queued() for tid in self.agents])
            mean_speed = np.mean([traci.vehicle.getSpeed(vid) for vid in all_vehs_in_env])
            return {
                "t": self.current_simulation_time,
                "wt": mean_waitting_time,
                "ql": mean_queue_length,
                "sd": mean_speed,
                # "b3c3_sd": traci.edge.getLastStepMeanSpeed("B3C3"),
                # "b3c3_ql": traci.edge.getLastStepHaltingNumber("B3C3")
            }

        # =======================
        # 以下都是旧实现
        # 感觉对于waitting_time的计算有误或是不严谨
        # 故弃之不用
        # =======================
        # total_stop = sum(self.traffic_signals[tid].get_total_queued() for tid in self.agents)
        # # 因为在计算info之前已经计算了reward self.traffic_signals[tid].last_measure 即是当前 waiting time
        # total_wait_time = sum(self.traffic_signals[tid].last_measure for tid in self.agents)
        # running_vehicles = traci.vehicle.getIDList()
        # total_speed = sum(traci.vehicle.getSpeed(vid) for vid in running_vehicles)  # type: ignore
        # every_traffic_signal_qlength = {tid: self.traffic_signals[tid].get_respective_queued() for tid in self.agents}
        # return {
        #     "step_time": self.current_simulation_time,
        #     "reward": sum(self.rewards[tid] for tid in self.agents),
        #     "total_stopped": total_stop,
        #     "total_wait_time": total_wait_time,
        #     "total_speed": total_speed,
        #     "avg_stopped": total_stop / len(self.agents),
        #     "avg_wait_time": total_wait_time / len(self.agents),
        #     "avg_speed": total_speed / len(running_vehicles) if len(running_vehicles) != 0 else 50,  # 50km/h 是默认最大速度
        #     "correlation_obs": (
        #         np.nanmean(
        #             [self.last_diff[ts]["correlation"]["obs"] for ts in self.agents]  # type: ignore
        #         )
        #         if self.cav_compare
        #         else "COMPARE OFF"
        #     ),
        #     # 比较cav环境下和全观测环境下的obs和reward(waits)的std差值
        #     "correlation_wait": (
        #         np.nanmean(
        #             [self.last_diff[ts]["correlation"]["waits"] for ts in self.agents]  # type: ignore
        #         )
        #         if self.cav_compare
        #         else "COMPARE OFF"
        #     ),
        #     "diff_reward": (
        #         np.mean(
        #             [self.last_diff[ts]["diffreward"]["result"] for ts in self.agents]  # type: ignore
        #         )
        #         if self.cav_compare
        #         else "COMPARE OFF"
        #     ),
        #     "every_traffic_signal_qlength": every_traffic_signal_qlength,
        #     "every_traffic_signal_action": self.last_action.copy(),
        #     "accident_position": self.accident_position,
        #     "C2_avg_stopped": self.traffic_signals["C2"].get_total_queued(),
        #     "C2_avg_speed": self.traffic_signals["C2"].get_avg_speed(),
        # }

    def close(self):
        traci.close()

    def save_csv(self, out_csv_name):
        df = pd.DataFrame(self.summary)
        print(f"save csv to: {out_csv_name}run{self.run}.csv")
        df.to_csv(f"{out_csv_name}run{self.run}.csv", index=False)

    def _load_neighbors(self):
        net = sumolib.net.readNet(self.net_file)
        nodes = net.getNodes()
        for node in nodes:
            if node.getID() in self.agents:
                neighbors = node.getNeighboringNodes()
                neighbor_set = set([n.getID() for n in neighbors if n.getID() in self.agents])
                self.neighbors[node.getID()] = neighbor_set

    def mixed_reward(self, center_tid: str):
        center_weight = 0.5
        neighbor_ids: set = self.neighbors[center_tid]
        neighbor_reward = sum([self.rewards[tid] for tid in neighbor_ids]) / len(neighbor_ids)
        return center_weight * self.rewards[center_tid] + (1 - center_weight) * neighbor_reward
