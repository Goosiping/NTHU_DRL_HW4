import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple
import random

def flatten_obs(obs):
    v = obs['v_tgt_field']
    v_flatten = v.flatten()

    new_obs = v_flatten
    new_obs = new_obs.astype(np.float)
    new_obs = np.append(new_obs, obs['pelvis']['height'])
    new_obs = np.append(new_obs, obs['pelvis']['pitch'])
    new_obs = np.append(new_obs, obs['pelvis']['roll'])
    new_obs = np.append(new_obs, obs['pelvis']['vel'])

    for leg in ['r_leg', 'l_leg']:
        new_obs = np.append(new_obs, obs[leg]['ground_reaction_forces'])
        for joint in ['hip_abd', 'hip', 'knee', 'ankle']:
            new_obs = np.append(new_obs, obs[leg]['joint'][joint])
            new_obs = np.append(new_obs, obs[leg]['d_joint'][joint])
        for muscle in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
            new_obs = np.append(new_obs, obs[leg][muscle]['f'])
            new_obs = np.append(new_obs, obs[leg][muscle]['l'])
            new_obs = np.append(new_obs, obs[leg][muscle]['v'])

    # print(new_obs)
    # print(new_obs.shape)
    # print(new_obs[0], new_obs[-1], type(new_obs[0]), type(new_obs[-1]))

    return new_obs

def init_weight(layer, mode='normal'):
    if mode == 'normal':
        nn.init.xavier_uniform_(layer.weight)
    if mode == 'uniform':
        nn.init.kaiming_normal_(layer.weight)

class ValueNetwork(nn.Module):
    def __init__(self, n_states, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim

        self.h1 = nn.Linear(self.n_states, self.hidden_dim)
        init_weight(self.h1)
        self.h1.bias.data.zero_()
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        init_weight(self.h2)
        self.h2.bias.data.zero_()
        self.h3 = nn.Linear(self.hidden_dim, 1)
        init_weight(self.h3)
        self.h3.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.h1(states))
        x = F.relu(self.h2(x))
        return self.h3(x)

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.h1 = nn.Linear(self.n_states + self.n_actions, self.hidden_dim)
        init_weight(self.h1)
        self.h1.bias.data.zero_()
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        init_weight(self.h2)
        self.h2.bias.data.zero_()
        self.h3 = nn.Linear(self.hidden_dim, 1)
        init_weight(self.h3)
        self.h3.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.h3(x)
    
class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.h1 = nn.Linear(self.n_states, self.hidden_dim)
        init_weight(self.h1)
        self.h1.bias.data.zero_()
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        init_weight(self.h2)
        self.h2.bias.data.zero_()
        self.h3 = nn.Linear(self.hidden_dim, self.n_actions)
        init_weight(self.h3)
        self.h3.bias.data.zero_()

        self.log_std = nn.Linear(self.hidden_dim, self.n_actions)
        init_weight(self.log_std)
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.h1(states))
        x = F.relu(self.h2(x))

        mean = self.h3(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = torch.distributions.Normal(mean, std)
        return dist
    
    def sample(self, states):
        dist = self(states)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u)

        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action.clamp(0, 1), log_prob

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class Memory:
    def __init__(self, memory_size):
        self.memory = []
        self.memory_size = memory_size

    def store(self, *transisiton):
        self.memory.append(Transition(*transisiton))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Agent:
    def __init__(self,n_states, n_actions, memory_size, batch_size, gamma, lr, tau, device):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.device = device
        
        # Memory and Networks
        self.memory = Memory(memory_size = self.memory_size)
        self.policy_net = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.q_net1 = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.q_net2 = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.value_net = ValueNetwork(self.n_states).to(self.device)
        self.value_target = ValueNetwork(self.n_states).to(self.device)
        self.value_target.load_state_dict(self.value_net.state_dict())
        self.value_target.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q_net1_opt = torch.optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_net2_opt = torch.optim.Adam(self.q_net2.parameters(), lr=self.lr)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def act(self, observation):
        state = flatten_obs(observation)
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(self.device)
        action, log_probs = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def store(self, state, action, reward, done, next_state):
        state = flatten_obs(state)
        next_state = flatten_obs(next_state)

        state = torch.from_numpy(state).float().to('cpu')
        action = torch.Tensor([action]).to('cpu')
        reward = torch.Tensor([reward]).to('cpu')
        done = torch.Tensor([done]).to('cpu')
        next_state = torch.from_numpy(next_state).float().to('cpu')
        self.memory.store(state, action, reward, done, next_state)

    def _unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

        return states, actions, rewards, dones, next_states

    def learn(self):

        if len(self.memory.memory) < self.batch_size:
            return 0, 0, 0
        
        # print("========== Start Learning ==========")
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = self._unpack(batch)

        # Value Loss
        re_actions, log_probs = self.policy_net.sample(states)
        q1 = self.q_net1(states, re_actions)
        q2 = self.q_net2(states, re_actions)
        q = torch.min(q1, q2)
        value_target = q.detach() - log_probs.detach()

        value = self.value_net(states)
        value_loss = self.value_loss(value, value_target)

        # Q Loss
        with torch.no_grad():
            target_q = rewards + self.gamma * self.value_target(next_states) * (1 - dones)
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        q1_loss = self.q_value_loss(q1, target_q)
        q2_loss = self.q_value_loss(q2, target_q)

        # Policy Loss
        policy_loss = (log_probs - q).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        self.q_net1_opt.zero_grad()
        q1_loss.backward()
        self.q_net1_opt.step()

        self.q_net2_opt.zero_grad()
        q2_loss.backward()
        self.q_net2_opt.step()

        self._soft_update(self.value_net, self.value_target)
        return value_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def _soft_update(self, learning_net, target_net):
        for target_param, learning_param in zip(target_net.parameters(), learning_net.parameters()):
            target_param.data.copy_(self.tau * learning_param.data + (1 - self.tau) * target_param.data)

    def save(self):
        torch.save(self.policy_net.state_dict(), '112062574_hw4_data')

        # torch.save(self.value_net.state_dict(), './checkpoints/value_learning.pth')
        # torch.save(self.value_target.state_dict(), './checkpoints/value_target.pth')
        # torch.save(self.q_net1.state_dict(), './checkpoints/q1_net.pth')
        # torch.save(self.q_net2.state_dict(), './checkpoints/q2_net.pth')


    def load(self):
        pass

if __name__ == "__main__":

    d = {'v_tgt_field':np.array([[[1.78051824,1.68897053,1.37599499,0.93186014,
    0.56084851,0.44584525,0.8415093,0.74154724,
    0.66103554,0.59527699,0.5408061],
    [1.5972991,1.54552244,1.34080287,0.96728491,
    0.55693485,0.32260857,0.60606907,0.52830385,
    0.46615182,0.4167047,0.3765211],
    [1.03211496,0.94373135,0.83088304,0.64244439,
    0.38370844,0.19709061,0.1647902,0.28772672,
    0.25203405,0.22415712,0.2017974],
    [0.11353006,0.11347769,0.09079594,0.07217856,
    0.04416696,0.02211573,0.01680792,0.030307,
    0.02646734,0.02349112,0.02111655],
    [-0.86476943,-0.78058401,-0.67073818,-0.52424902,
    -0.31571526,-0.16037631,-0.12917561,-0.22897275,
    -0.20034584,-0.17805003,-0.16020189],
    [-1.47583118,-1.43056698,-1.25742236,-0.92345915,
    -0.53491155,-0.29693544,-0.29345314,-0.47459699,
    -0.41792494,-0.37306708,-0.33674562],
    [-1.79856081,-1.71414409,-1.41539972,-0.9681627,
    -0.56724308,-0.41143481,-0.79047203,-0.69480506,
    -0.6177893,-0.5552997,-0.50378748],
    [-1.48519862,-1.39299148,-1.10592066,-0.7648428,
    -0.56549734,-1.11674365,-0.99065395,-0.88369774,
    -0.79479294,-0.72047614,-0.65783677],
    [-0.93502036,-0.88963856,-0.77146438,-0.69686144,
    -1.40328618,-1.27174219,-1.148706,-1.04072197,
    -0.94720588,-0.86653493,-0.79686185],
    [-0.84875948,-0.85807394,-0.90758905,-1.6154145,
    -1.50128896,-1.38410484,-1.27192055,-1.16872954,
    -1.07588707,-0.99329757,-0.92018246],
    [-1.79973327,-1.78504094,-1.7356178,-1.65949423,
    -1.56690637,-1.46724773,-1.36749924,-1.27204713,
    -1.18320585,-1.10188875,-1.02816702]],
    [[0.06185974,-0.48318886,-0.83514744,-0.86465361,
    -0.70048476,-0.69998551,-1.59112497,-1.64010442,
    -1.67418093,-1.69867822,-1.71679992],
    [0.08170276,-0.65121623,-1.19848982,-1.32172649,
    -1.02429902,-0.74586866,-1.68555072,-1.72067208,
    -1.73854503,-1.75105943,-1.76014114],
    [0.10013726,-0.75398741,-1.40827599,-1.66451039,
    -1.3380457,-0.86395835,-0.87025201,-1.77679964,
    -1.78221918,-1.78594459,-1.78861317],
    [0.10586519,-0.87180941,-1.47992447,-1.79812441,
    -1.48055326,-0.93189041,-0.85362948,-1.79968915,
    -1.79975635,-1.7998029,-1.79983658],
    [0.10586519,-0.78746849,-1.43551793,-1.71525494,
    -1.39035952,-0.88774774,-0.86100828,-1.78532374,
    -1.7887682,-1.79112961,-1.79281803],
    [0.08477254,-0.67718245,-1.26274834,-1.41771399,
    -1.10528072,-0.77111387,-0.91572529,-1.73625756,
    -1.75076701,-1.76087472,-1.76818332],
    [0.06747512,-0.53004249,-0.9284879,-0.9708806,
    -0.76554935,-0.69784458,-1.61511808,-1.66045265,
    -1.69062159,-1.71216686,-1.72802753],
    [0.04136065,-0.31981285,-0.53863771,-0.56944493,
    -0.56660803,-1.40687758,-1.50282359,-1.56810587,
    -1.61498806,-1.64948521,-1.67545319],
    [0.02070559,-0.16242453,-0.29879608,-0.41258667,
    -1.11866002,-1.27379575,-1.38577184,-1.46860158,
    -1.53058722,-1.57766186,-1.61397478],
    [0.01561226,-0.13004947,-0.29177685,-0.79391368,
    -0.99298086,-1.1507149,-1.27362333,-1.36893182,
    -1.44304319,-1.50108947,-1.54698958],
    [0.02828677,-0.23123702,-0.47695313,-0.69709976,
    -0.88582643,-1.04263123,-1.1704059,-1.27350149,
    -1.35644314,-1.42329141,-1.47742766]]]),
    'pelvis':{'height':0.9410979407136865,
    'pitch':-0.010442748417726485,
    'roll':-1.62250738510785e-05,
    'vel':[0.3549708439548275,0.0023519683954684066,0.07903808166905908,-0.9268499016264509,0.009237032292035123,-0.01629485939717691]},
    'r_leg':{'ground_reaction_forces':[-0.11909536444073085,0.05363998632973335,0.467757673103764],
    'joint':{'hip_abd':-0.00021766773969338088,
    'hip':0.048824878607440225,
    'knee':0.0642165722498084,
    'ankle':0.01954361896270135},
    'd_joint':{'hip_abd':-0.0025997771840570423,
    'hip':4.455747629294257,
    'knee':5.889713478980696,
    'ankle':2.032150280617952},
    'HAB':{'f':0.20667832738601552,'l':0.8969599554544039,'v':-1.050678975747996},
    'HAD':{'f':0.29133634049979173,'l':0.6236305878362097,'v':0.06334179429629888},
    'HFL':{'f':0.49153019139202153,'l':1.1057800281630903,'v':1.2312672146008061},
    'GLU':{'f':0.1981781483903978,'l':1.0036486639057862,'v':-1.6574123393091147},
    'HAM':{'f':0.10002813150138733,'l':0.8886192989075864,'v':-4.0711010884929815},
    'RF':{'f':0.18217218964445062,'l':0.7273868846707132,'v':-2.798638334423388},
    'VAS':{'f':0.11836298609526698,'l':0.7586908874868036,'v':-3.440927793638186},
    'BFSH':{'f':0.7446867812266008,'l':1.208067440214152,'v':0.1755160165583627},
    'GAS':{'f':0.11975270197054219,'l':1.0474041847935454,'v':-3.1627146516138565},
    'SOL':{'f':0.058848757234943656,'l':0.9928033221858861,'v':-1.4604538811655632},
    'TA':{'f':0.28022238589327986,'l':0.8860704594750464,'v':-0.513354866686928}},
    'l_leg':{'ground_reaction_forces':[0.12214082735814046,0.07197211265145272,0.33035904032431745],
    'joint':{'hip_abd':-3.6297836721825176e-05,
    'hip':0.05811352108239795,
    'knee':0.0816157386756189,
    'ankle':0.027000207625414007},
    'd_joint':{'hip_abd':-0.04516102846620927,
    'hip':4.5832604686354435,
    'knee':5.734344724091904,
    'ankle':0.31703683621178713},
    'HAB':{'f':0.1125192304575424,'l':0.897772970389893,'v':-1.4132990724097945},
    'HAD':{'f':0.2720284054816946,'l':0.6244642573055867,'v':-0.18984023483087775},
    'HFL':{'f':0.5680086506662563,'l':1.1063701590319848,'v':0.8588847764326805},
    'GLU':{'f':0.28166565678610916,'l':0.9993089140563899,'v':-1.7473142892756037},
    'HAM':{'f':0.2120569998449774,'l':0.8577337376173412,'v':-3.125655839369149},
    'RF':{'f':0.088315885246342,'l':0.7505186200017405,'v':-4.930547005673515},
    'VAS':{'f':0.06693807916321583,'l':0.7557254637269578,'v':-2.5955096481402102},
    'BFSH':{'f':0.5267496271837093,'l':1.2166807444453225,'v':0.9484421886955086},
    'GAS':{'f':0.08271267962765141,'l':1.0933010523676037,'v':-2.7346651629954986},
    'SOL':{'f':0.10367513780929254,'l':0.9407990152181027,'v':-3.4246175694829666},
    'TA':{'f':0.40072973681481705,'l':0.8786901489088115,'v':-1.300647964716506}}
    }
  
    # s = flatten_obs(d)
    # print(s.shape)
    # print(type(s[0]), s[0], type(s[-1]), s[-1])

    # test_value = ValueNetwork(n_states=s.shape[0])
    # test_q = QNetwork(n_states=s.shape[0], n_actions=22)
    # test_policy = PolicyNetwork(n_states=s.shape[0], n_actions=22)

    # action, log_probs = test_policy.sample(torch.tensor(s).float())
    # print("PolicyNetwork:", end='')
    # print(action)
    # print("ValueNetwork:", end='')
    # print(test_value(torch.tensor(s).float()))
    # print("QNetwork:", end='')
    # s = torch.tensor(s).float().unsqueeze(0)
    # action = action.unsqueeze(0)
    # print(action.shape, s.shape)
    # print(test_q(s, action))


