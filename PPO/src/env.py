"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import gym
import retro
# import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
# from nes_py.wrappers import JoypadSpace
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import ffmpeg
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp
from src.helpers import JoypadSpace, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get
from src.retrowrapper import RetroWrapper
from retinavision.cortical_functions.cortical_map_image import CorticalMapping
from matplotlib import pyplot as plt

SCRIPT_DIR = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
ENV_NAME = 'SMB-JU'
LVL_ID = 'Level1-1'

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())
    
    # def vidwrite(self, images, framerate=60, vcodec='libx264'):
    #     if not isinstance(images, np.ndarray):
    #         images = np.asarray(images)
    #     n,height,width,channels = images.shape
    #     process = (
    #         ffmpeg
    #             .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    #             .output(saved_path, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
    #             .overwrite_output()
    #             .run_async(pipe_stdin=True)
    #     )
    #     for frame in images:
    #         process.stdin.write(
    #             frame
    #                 .astype(np.uint8)
    #                 .tobytes()
    #         )
    #     process.stdin.close()
    #     process.wait()


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if flag_get(info): #info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)

class CorticalPreprocessor(Wrapper):
    def __init__(self, env, cortex_left, cortex_right, retina_resolution):
        super(CorticalPreprocessor, self).__init__(env)
        self.cortical_mapper = CorticalMapping()
        self.cortical_mapper.setup_retina(retina_resolution=retina_resolution)
        self.cortical_mapper.setup_cortex(cortex_left=cortex_left, cortex_right=cortex_right)
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        transformed_state = self.cortical_mapper.cortical_transform(state)
        return transformed_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        transformed_state = self.cortical_mapper.cortical_transform(state)
        return transformed_state
        
def create_train_env(actions, cortex_left, cortex_right, retina_resolution, output_path=None, mp_wrapper=True, save_video=False, use_retina=True):
    # env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
    print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    obs_type = retro.Observations.IMAGE # or retro.Observations.RAM

    if mp_wrapper:
        env = RetroWrapper(ENV_NAME, state=LVL_ID, record=save_video, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
    else:
        env = retro.make(ENV_NAME, LVL_ID, record=save_video, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)

    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)

    # Activate cortical transformation
    if use_retina:
        env = CorticalPreprocessor(env, cortex_left=cortex_left, cortex_right=cortex_right, retina_resolution=retina_resolution)

    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env

def create_train_env_test(actions, output_path=None, mp_wrapper=True, save_video=False):
    # env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "retro_integration"))
    print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    print(ENV_NAME in retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))
    obs_type = retro.Observations.IMAGE # or retro.Observations.RAM

    if mp_wrapper:
        env = RetroWrapper(ENV_NAME, state=LVL_ID, record=save_video, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)
    else:
        env = retro.make(ENV_NAME, LVL_ID, record=save_video, inttype=retro.data.Integrations.CUSTOM_ONLY, obs_type=obs_type)

    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)

    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env


class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, cortex_left, cortex_right, retina_resolution, use_retina, output_path=None, save_video=False):
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT

        # self.envs = create_train_env(actions, output_path=output_path)
        self.envs = [create_train_env(actions, output_path=output_path, save_video=save_video, cortex_left=cortex_left, cortex_right=cortex_right, retina_resolution=retina_resolution, use_retina=use_retina) for _ in range(num_envs)]
        
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        self.num_envs = len(self.envs)

        # for index in range(num_envs):
        #     process = mp.Process(target=self.run, args=(index,))
        #     process.start()
        #     self.env_conns[index].close()

    # def run(self, index, actions=None, output_path=None):
    #     if actions is not None:
    #         env = create_train_env(actions, output_path=output_path)
    #         self.envs.append(env)
    #     else:
    #         self.agent_conns[index].close()
    #         while True:
    #             request, action = self.env_conns[index].recv()
    #             if request == "step":
    #                 self.env_conns[index].send(self.envs[index].step(action.item()))
    #             elif request == "reset":
    #                 self.env_conns[index].send(self.envs[index].reset())
    #             else:
    #                 raise NotImplementedError
