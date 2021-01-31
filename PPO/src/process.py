"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
import numpy as np
from src.helpers import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, flag_get
import csv, os
import time
import cv2
from retinavision.cortical_functions.cortical_map_image import CorticalMapping
from matplotlib import pyplot as plt
from skimage.transform import resize

def evaluate(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT

    savefile = opt.saved_path + '/PPO_test.csv'
    print(savefile)
    title = ['Steps', 'Time', 'TotalReward', "Flag"]
    with open(savefile, 'w', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    print(opt.retina_resolution)
    env = create_train_env(actions, mp_wrapper=False, cortex_left=opt.cortex_left, cortex_right=opt.cortex_right, retina_resolution=opt.retina_resolution, use_retina=opt.retina)

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()

    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    
    done = True
    curr_step = 0
    tot_step = 0
    actions = deque(maxlen=opt.max_actions)
    tot_reward = 0
    got_flag = 0
    index = 0
    while True:
        start_time = time.time()
        curr_step += 1
        tot_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())

        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item() # This selects the best action to take
        state, reward, done, info = env.step(action)


        # im1 = state[0, 0, :, :]
        # im2 = state[0, 1, :, :]
        # im3 = state[0, 2, :, :]
        # im4 = state[0, 3, :, :]

        # res1 = cv2.resize(im1, dsize=(370, 370), interpolation=cv2.INTER_CUBIC)
        # im2 = state[0, 1, :, :]
        # res2 = cv2.resize(im2, dsize=(370, 370), interpolation=cv2.INTER_CUBIC)
        # im3 = state[0, 2, :, :]
        # res3 = cv2.resize(im2, dsize=(370, 370), interpolation=cv2.INTER_CUBIC)
        # im4 = state[0, 3, :, :]
        # res4 = cv2.resize(im2, dsize=(370, 370), interpolation=cv2.INTER_CUBIC)


        # fig=plt.figure(figsize=(8, 8))
        # columns = 2
        # rows = 2
        # fig.add_subplot(rows, columns, 1)
        # plt.imshow(im1)
        # fig.add_subplot(rows, columns, 2)
        # plt.imshow(im2)
        # fig.add_subplot(rows, columns, 3)
        # plt.imshow(im3)
        # fig.add_subplot(rows, columns, 4)
        # plt.imshow(im4)
        # plt.show()
        
        index += 1
        tot_reward += reward

        # Uncomment following lines if you want to save model whenever level is completed
        if flag_get(info):
            print("Evaluate: Level Completed!")
            got_flag = 1
            done = True
            torch.save(local_model.state_dict(),
                       "{}/ppo_super_mario_bros_{}".format(opt.saved_path, curr_step))

        # env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            # print("Evaluate: Time's up!")
            done = True

        if done:
            # print("Evaluate: Done!")
            ep_time = time.time() - start_time
            data = [tot_step, "{:.4f}".format(ep_time), "{:.2f}".format(tot_reward), got_flag]
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            curr_step = 0
            got_flag = 0
            tot_reward = 0
            actions.clear()
            # time.sleep(10) # Sleep for 10 secs
            state = env.reset()

        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
