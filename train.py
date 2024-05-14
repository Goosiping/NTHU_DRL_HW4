from osim.env import L2M2019Env
import numpy as np
import wandb
import time
from Agent import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# wandb
# wandb.init(project="nthu-drl-hw4")

# Parameters
episodes = 10000
checkpoint = 100
memory_size = 4000000
batch_size = 256
gamma = 0.99
lr = 0.0005
tau = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Environment
    env = L2M2019Env(visualize=False)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    # Agent
    agent = Agent(
        n_states=n_states, 
        n_actions=n_actions, 
        memory_size=memory_size, 
        batch_size=batch_size, 
        gamma=gamma, 
        lr=lr, 
        tau=tau,
        device=device
    )

    # Start Training
    rewards = []
    start_time = time.time()
    for episode in range(episodes):

        state = env.reset()
        total_reward = 0
        done = False

        # Start an episode
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, done, next_state)
            value_loss, q_loss, policy_loss = agent.learn()
            total_reward += reward
            state = next_state

        # Episode terminated
        rewards.append(total_reward)
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Value Loss', value_loss, episode)
        writer.add_scalar('Q Loss', q_loss, episode)
        writer.add_scalar('Policy Loss', policy_loss, episode)
        print("Episode: {:4d} | Total Reward: {:.2f} | Value Loss: {:.2f} | Q Loss: {:.2f} | Policy Loss: {:.2f} | Time: {:.2f}".format(
            episode, total_reward, value_loss, q_loss, policy_loss, time.time() - start_time
        ))
        if episode % checkpoint == 0:
            agent.save()
            # with open("logs.txt", "a") as f:
            #     f.write(f"{total_reward}\n")
            # wandb.log({"Total Reward": total_reward, "Time": time.time() - start_time})
         
        
        