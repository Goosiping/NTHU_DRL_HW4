from osim.env import L2M2019Env

if __name__ == "__main__":
    env = L2M2019Env(visualize=True)
    observation = env.reset()
    print("initial observation:", observation)
    print(type(observation['v_tgt_field']))