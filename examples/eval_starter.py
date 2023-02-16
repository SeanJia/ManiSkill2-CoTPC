import numpy as np
from tqdm import tqdm
import torch
import gym
import os
import argparse
import mani_skill2.envs  # Required to load the maniskill2 environments.

USER_ROOT = ''  ##### Specify your base path #####


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='LiftCube-v1')
    parser.add_argument('--num_traj_to_eval', type=int, default=50)
    parser.add_argument('--obs_mode', type=str, default='state')
    parser.add_argument('--control_mode', type=int, default='')

    return parser.parse_args()
    

def predict(model, *args):
    ##### YOUR IMPLEMENTATION HERE #####
    # You might need torch.from_numpy(...).float()
    pass


if __name__ == "__main__":

    args = parse_args()
    if not args.control_mode:
        print('Please specify --control_mode.')

    env = gym.make(args.task, obs_mode=args.obs_mode, control_mode=args.control_mode)

    ##### YOUR IMPLEMENTATION HERE TO LOAD THE MODEL #####
    model = ''    
    model.eval()

    success_rates = []

    # In our evaluation, we will override the max_step=200 set for each env.
    if args.task.startswith('LiftCube'):  
        max_steps = 100
    if args.task.startswith('StackCube'):  
        max_steps = 200
    if args.task.startswith('PegInsertionSide'):  
        max_steps = 250
    if args.task.startswith('TurnFaucet'):  
        max_steps = 250            
    if args.task.startswith('PushChair'):  
        max_steps = 300 
    else:
        assert False, f'Unknown env: {args.task}'
    
    for seed in tqdm(range(args.num_traj_to_eval)):  # Fix seed for evaluation.
        curr_success_rates = []
        env.seed(seed)
        state = env.reset()  

        for _ in range(max_steps):
            action = predict(model, state)
            state, _, _, info = env.step(action)
            curr_success_rates.append(info['success'])

        # Here we define a trajectory to be successful if at any moment it succeeds.
        success_rates.append(np.any(curr_success_rates))
    
    print(f'The overall success rate for {args.task} is: {np.mean(success_rates)}.')
