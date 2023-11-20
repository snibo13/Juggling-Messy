import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


class JugglingEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is a bimanual juggling environment powered by teh Mujoco physics simulator
    This environment involves a pair of RRR arms that are moving to catch and toss a single ball
    The arms have three independent actuators along the x and y axis.

    ## Action space
    The agent takes a 3-element vector for actions
    The space is continuous in [-1,1] where action represents a normalized torque


    ## Observation space
    The position of the end effector
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(-np.inf, np.inf, shape=(19,), dtype=np.float64)
        xml_path = os.path.join(os.path.dirname(__file__), "RR.xml")
        MujocoEnv.__init__(
            self, xml_path, frame_skip=2, observation_space=observation_space, **kwargs
        )
        num_joints = 3
        self.action_space = Box(
            -1, 1, shape=(num_joints,), dtype=np.float64
        )  # 2D action space, normalized to [-1, 1]

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

        self.ball_state = "air"

        self.init_qpos = self._get_obs()[: self.observation_structure["qpos"]].copy()
        self.init_qvel = self._get_obs()[: self.observation_structure["qvel"]].copy()

        # print(self.init_qpos)
        # print(self.init_qvel)

    def step(self, a):
        reward_ctrl = -np.square(a).sum()
        target = self.get_body_com("ball")
        vec = self.get_body_com("ee") - target

        # if self.ball_state == "air":
        reward_dist = -np.linalg.norm(vec)
        # if self.ball_state == "catch":
        #     reward_dist = 0
        #     reward_dist = -5 + np.linalg.norm(vec)

        gamma = 0.1
        reward = reward_dist + gamma * reward_ctrl
        # reward = 0

        # print("Action", a)
        self.do_simulation(a, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        truncated = False

        # Terminate if z value is lower than 0
        terminated = target[2] < -1  # Floor at -0.5
        if terminated:
            reward -= 10

        caught = False
        # Update ball state
        if np.linalg.norm(vec) < 0.15:
            reward += 1000
            # reward += abs(target[2] * 100)  # Reward for catching the ball higher
            caught = True
            terminated = True

        # if self.ball_state == "catch" and np.linalg.norm(vec) > 0.2:
        #     self.ball_state = "air"

        # terminated = False
        return (ob, reward, terminated, truncated, {"caught": caught})

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def _get_obs(self):
        return np.hstack([self.data.qpos, self.data.qvel])

    def reset_model(self, seed=None, options=None):
        qpos = self.init_qpos

        # noise = np.random.uniform(low=-0.05, high=0.05, size=qpos.size)
        # height = np.random.uniform(low=1.5, high=2)
        x = np.random.uniform(low=-0.05, high=0.05)
        y = np.random.uniform(low=-0.0, high=0.05)
        # qpos += noise
        qpos[3] = np.random.uniform(low=-0.4, high=0.4)
        qpos[4] = np.random.uniform(low=0.4, high=0.8)
        qpos[5] = np.random.uniform(low=1.4, high=1.8)

        # print(qpos)
        # exit()
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return (self._get_obs(), {})

    def reset(self, seed=None, options=None):
        return self.reset_model(seed, options)
