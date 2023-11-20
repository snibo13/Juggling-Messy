import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os


class BimanualEnv(MujocoEnv, utils.EzPickle):
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
        observation_space = Box(-np.inf, np.inf, shape=(31 - 3,), dtype=np.float64)
        xml_path = os.path.join(os.path.dirname(__file__), "Bimanual.xml")
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

        self.init_qpos = self._get_obs()[: self.observation_structure["qpos"]].copy()
        self.init_qvel = self._get_obs()[: self.observation_structure["qvel"]].copy()

        self.catching = "left"
        self.thrown = False
        self.caught = False
        self.apex = np.array([-0.75, 0.9, 1.5])
        self.catches = 0

        self.left_hand = self.get_body_com("palm_l")
        self.right_hand = self.get_body_com("palm")

    def step(self, a):
        assert not np.isnan(a).any()
        ball = self.get_body_com("ball")
        palm_l = self.get_body_com("palm_l")
        palm = self.get_body_com("palm")
        palm_offset = np.array([0, 0, 0.2])
        ball_vec_left = palm_l + palm_offset - ball
        ball_vec_right = palm + palm_offset - ball
        ee_displacement = palm - self.right_hand
        ee_l_displacement = palm_l - self.left_hand

        caught = False
        thrown = False

        reward = 0

        if self.catching == "left":
            vec_catch = ball_vec_left
            vec_throw = ball_vec_right
        else:
            vec_catch = ball_vec_right
            vec_throw = ball_vec_left

        palm_index = self.model.body("palm_l").id
        ball_index = self.model.body("ball").id

        # Desired behaviour:
        # 1. Catch the ball
        # a Minimize distance between ball and catching hand
        # b Explicitly reward the ball being in the catching hand
        # 2. Throw the ball
        # a Maximize distance between ball and throwing hand
        # b Explicitly reward the ball being out of the throwing hand
        # c Penalize the ball being in the throwing hand
        # d Penalize movement of the throwing hand
        # e Reward high ball height
        # f Reward proximity to initial catching hand position

        reward_1b = 0
        for contact in self.data.contact:
            if (
                palm_index == contact.geom1
                and ball_index == contact.geom2
                or palm_index == contact.geom2
                and ball_index == contact.geom1
            ) and not self.caught:
                print("Caught")
                caught = True
                self.caught = True
                self.thrown = False
                self.catching = "right" if self.catching == "left" else "left"
                reward_1b = 1000

        if (
            not caught
            and np.linalg.norm(vec_throw) > 0.3
            and self.caught
            and not self.thrown
            and self.model.body("ball").pos[2] > 0.2
        ):
            thrown = True
            self.thrown = True
            self.caught = False

            reward += 1000 if thrown else 0

        reward_1a = -np.linalg.norm(vec_catch)

        reward_2a = self.thrown * np.linalg.norm(vec_throw)
        reward_2b = (1000 * self.thrown) if np.linalg.norm(vec_throw) > 0.3 else 0
        reward_2c = -100 if np.linalg.norm(vec_throw) < 0.15 or ball[2] < 0.5 else 0
        reward_2d1 = -np.linalg.norm(ee_displacement)
        reward_2d2 = -np.linalg.norm(ee_l_displacement)
        reward_2e = ball[2]
        reward_2f = -np.linalg.norm(
            (ball - self.left_hand)
            if self.catching == "right"
            else (ball - self.right_hand)
        )

        reward_ctrl = -np.square(a).sum()

        reward += (
            reward_1a  # Catch proximity reward
            + reward_1b  # Instantaneous reward for catching
            # + reward_2a   # Throw distance reward
            # + reward_2b   # Throw instantaneous reward
            # + reward_2c   # Penalty for holding the ball
            + reward_2d1  # Reward for staying close to the initial position
            + reward_2d2  # Reward for staying close to the initial position
            # + reward_2e   # Reward for high ball height
            # + reward_2f   # Reward for throw proximity to initial catching hand position
            + reward_ctrl  # Penalty for action magnitude
        )

        # simulation and termination conditions
        self.do_simulation(a, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        truncated = False

        terminated = ball[2] < -1
        if terminated:
            reward -= 2000

        return (
            ob,
            reward,
            terminated,
            truncated,
            {"caught": caught, "thrown": thrown},
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def _get_obs(self):
        return np.hstack([self.data.qpos, self.data.sensordata, self.data.qvel])

    def reset_model(self, seed=None, options=None):
        qpos = self.init_qpos
        qvel = self.init_qvel

        qvel = np.zeros_like(qvel)

        self.set_state(qpos, qvel)
        self.catching = "left"
        self.caught = False
        self.thrown = False
        self.apex = np.array([-1.5 / 2.0, 0.5, 2])
        return (self._get_obs(), {})

    def reset(self, seed=None, options=None):
        return self.reset_model(seed, options)
