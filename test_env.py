import gym
from matplotlib import pyplot as plt
import mani_skill2.envs

env = gym.make("PegInsertionSide-v0", obs_mode=None, control_mode="pd_joint_delta_pos")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    #env.render(mode='human')  # a display is required to render
    #img = env.render(mode="cameras")
    #plt.savefig('/home/zjia/Research/inter_seq/test.png')
    # plt.imshow(img)
env.close()
