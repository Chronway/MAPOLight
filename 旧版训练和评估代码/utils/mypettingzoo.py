# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:01:07 2022

@author: Wang Chong
"""
from gymnasium import Env
from gymnasium.core import ObsType, ActType

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from gymnasium.spaces import Dict


@PublicAPI
class MyPettingZooEnv(MultiAgentEnv):
    """An interface to the PettingZoo MARL environment library.

    See: https://github.com/Farama-Foundation/PettingZoo

    Inherits from MultiAgentEnv and exposes a given AEC
    (actor-environment-cycle) game from the PettingZoo project via the
    MultiAgentEnv public API.

    Environments are positive sum games (-> Agents are expected to cooperate
    to maximize reward). This isn't a hard restriction, it just that
    standard algorithms aren't expected to work well in highly competitive
    games.

    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        env.reset()

        self.action_space = Dict(self.env.action_spaces)
        self.observation_space = Dict(self.env.observation_spaces)
        self._agent_ids = set(self.env.agents)

        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

    @property
    def unwrapped(self):
        return self.env

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        # print("reset:", self.env.observe(self.env.agent_selection).shape)
        return {self.env.agent_selection: self.env.observe(self.env.agent_selection)}, {}

    def step(self, action):
        self.env.step(action[self.env.agent_selection])
        obs_d = {}
        rew_d = {}
        terminated_d = {}
        truncated_d = {}
        info_d = {}

        while self.env.agents:
            obs, rew, terminated, truncated, info = self.env.last()
            a = self.env.agent_selection
            obs_d[a] = obs
            rew_d[a] = rew
            terminated_d[a] = terminated
            truncated_d[a] = truncated
            info_d[a] = info
            if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
                self.env.step(None)
            else:
                break

        all_done = not self.env.agents
        terminated_d["__all__"] = all_done
        truncated_d["__all__"] = all_done
        # print("step:", obs_d[self.env.agent_selection].shape)
        return obs_d, rew_d, terminated_d, truncated_d, info_d

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        return self.env.render(mode)

    @property
    def get_sub_environments(self):
        return self.env.unwrapped
