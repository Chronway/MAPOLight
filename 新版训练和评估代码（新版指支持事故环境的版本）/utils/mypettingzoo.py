# -*- coding: utf-8 -*-
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from gymnasium.spaces import Dict

@PublicAPI
class MyPettingZooEnv(MultiAgentEnv):
    """
    rllib不直接支持pettingzoo环境，故需要用该包装器将pettingzoo环境包装成rllib的MultiAgentEnv环境
    事实上rllib提供了一个pettingzoo环境包装器，但那个包装器存在多余的限制，且unwrapped实现不规范，故在此我手动实现一个包装器
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = Dict(self.env.unwrapped.action_spaces)
        self.observation_space = Dict(self.env.unwrapped.observation_spaces)
        self._agent_ids = set(self.env.unwrapped.possible_agents)

        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        # print("reset:", self.env.observe(self.env.agent_selection).shape)
        return ({self.env.agent_selection: self.env.observe(self.env.agent_selection)},
                {self.env.agent_selection: self.env.infos[self.env.agent_selection]})

    def step(self, action: dict):
        self.env.step(action[self.env.agent_selection])
        agent = self.env.agent_selection
        obs, rew, terminated, truncated, info = self.env.last()
        observation = {agent: obs}
        reward = {agent: rew}
        terminated = {agent: terminated}
        truncated = {agent: truncated}
        info = {agent: info}

        # 按照rllib的奇葩标准，必须显式添加__all__
        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())

        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    @property
    def sub_env(self):
        return self.env

