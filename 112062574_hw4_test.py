from osim.env import L2M2019Env
from Agent import flatten_obs, PolicyNetwork
import torch
import numpy as np

class Agent(object):
    def __init__(self):
        self.policy = PolicyNetwork(339, 22)
        self.policy.load_state_dict(torch.load('112062574_hw4_data', map_location=torch.device('cpu')))
        self.policy.eval()
        self.device = 'cpu'
        
    def act(self, observation):
        state = flatten_obs(observation)
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(self.device)
        action, log_probs = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
if __name__ == "__main__":
    env = L2M2019Env(visualize=True)
    observation = env.reset()

    agent = Agent()

    total_reward = 0
    done = False
    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward)


