import numpy as np

mouse_act = np.array([3.99846304e-04, 9.62694339e-05, 9.25386742e-05, 2.35932772e-04,
       5.99540492e-05, 9.73042044e-05, 9.43439562e-05, 3.55796057e-04,
       7.35257315e-05, 8.76583687e-05, 7.79009626e-05, 2.75649840e-04,
       6.77305141e-05, 5.48222892e-05, 8.95565191e-05, 7.24655426e-05,
       9.56808268e-05, 1.45156781e-04, 3.87784597e-05, 4.69305679e-05,
       8.51980489e-05, 1.89058718e-04, 1.06683393e-04, 9.27377235e-05,
       6.43162118e-05, 6.18879197e-05, 1.58092172e-04, 7.19337479e-05,
       6.81892252e-05, 6.61466256e-05, 1.86970985e-04, 3.06520313e-04,
       1.09735959e-04, 1.25378932e-04, 6.06179245e-05, 5.23850820e-05,
       1.32768740e-04, 2.97313255e-04, 9.33151109e-05, 5.19627727e-05,
       4.43425674e-05, 2.48651306e-04, 6.24462603e-05, 1.07035246e-04,
       1.51836448e-04, 1.02070088e-04, 8.53040766e-05, 1.76727266e-04,
       4.86363184e-05, 6.83379991e-05, 8.39406913e-05, 8.30375187e-05,
       8.16107744e-05, 8.48478446e-05, 2.38247720e-04, 6.84288351e-05,
       6.56022306e-05, 5.70556151e-05, 6.66112376e-05, 4.57564222e-05,
       1.10231882e-04, 1.09025471e-04, 7.45720294e-05, 5.66215272e-05,
       1.78961640e-04, 9.36763142e-05, 1.00799858e-04, 4.68466257e-05,
       4.96205448e-05, 1.28723082e-04, 5.56249690e-05, 6.33580283e-05,
       3.72465519e-05, 7.75216279e-05, 7.57825621e-05, 1.69878924e-04,
       1.47108130e-04, 6.46117064e-05, 4.83975470e-05, 5.08284263e-05,
       1.27804923e-04, 1.17060386e-04, 4.98564864e-05, 1.41718761e-04,
       2.80153625e-05, 2.55310585e-05, 3.49115549e-05, 2.10367727e-05,
       1.43210997e-05, 2.41098144e-05, 1.38213082e-05, 1.12978276e-05,
       1.26056907e-05, 2.89524934e-05, 3.75236893e-05, 2.51866617e-05,
       5.06926408e-05, 3.64422737e-05, 3.47950060e-05, 4.33088258e-05,
       2.26869546e-05, 1.91999470e-05, 1.38305886e-05, 3.23035719e-05,
       1.51757632e-05, 3.98517862e-05, 9.39725145e-05, 1.97420248e-05,
       1.59942052e-05, 1.22270527e-05, 2.12940680e-05, 1.35680856e-05,
       2.10970888e-05, 1.62339806e-05, 1.53805875e-05, 3.08837390e-05,
       1.26314369e-05, 1.31810458e-05, 1.26858119e-05, 7.59617307e-05,
       1.28127208e-05, 1.17330583e-05, 1.89725143e-05, 1.96602841e-05,
       2.87700259e-05, 3.51415862e-05, 3.35940013e-05, 1.79192407e-05,
       2.18944506e-05, 2.07641242e-05, 2.06900136e-05, 2.46415646e-05,
       2.41177929e-05, 1.48340908e-05, 4.32405137e-05, 2.77315256e-05,
       2.07095438e-05, 4.09299754e-05, 1.84330108e-05, 1.62990181e-05,
       1.77133679e-05, 1.32260168e-04, 1.35006416e-05, 2.90336188e-05,
       2.44666646e-05, 2.51260286e-05, 5.58975174e-05, 4.71327198e-05,
       3.96816393e-05, 1.36088508e-04, 3.74423749e-05, 4.21947620e-05,
       2.47290779e-05, 2.49955508e-05, 1.35984237e-05, 2.18590405e-05,
       2.28138173e-05, 1.37822477e-05, 2.98004465e-05, 5.05085745e-05,
       1.78483594e-05, 3.16405845e-05, 2.44751077e-05, 1.47911667e-05,
       6.61268816e-05, 3.26353738e-05, 4.43642050e-05, 5.08070514e-05,
       3.13836689e-05, 3.12767524e-05, 4.01920041e-05, 1.47742871e-05,
       1.01736213e-04, 9.38884494e-05, 2.50671531e-05, 2.88339669e-05,
       1.15461322e-05, 1.33230461e-05, 3.28351937e-05, 3.47075954e-05,
       1.31742455e-05, 2.89207427e-05, 1.23978995e-05, 1.23687065e-05,
       3.27064844e-05, 7.55675959e-05, 1.29350682e-05, 3.58008418e-05,
       5.60248557e-05, 1.65076732e-05, 1.27520003e-05, 2.65878662e-05,
       1.82358370e-05, 1.78290067e-05, 3.21014011e-05, 3.82615221e-05,
       2.75355671e-05, 1.33211409e-05, 3.99211419e-05, 5.25600728e-05,
       1.56000467e-05, 2.31559321e-05, 1.73586031e-05, 2.02382587e-05,
       3.01775184e-05, 8.72756418e-05, 3.00132859e-05, 1.92504017e-05,
       1.29799410e-05, 2.19124299e-05, 4.63171865e-05, 1.00591637e-04,
       1.25173115e-04, 2.49372027e-04, 3.90651919e-04, 8.46775605e-05,
       5.99837715e-04, 9.44390882e-05, 9.29237692e-05])

# print(act)

import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 219)
        self.fc2 = nn.Linear(219, 219)
        self.fc3 = nn.Linear(219, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def brain_act(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(use_mouse_act=False):
    env = gym.make('CartPole-v1')
    q = Qnet()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    scores = []
    if use_mouse_act:
        s, _ = env.reset()
        for _ in range(1000):
            # print(q.brain_act(torch.from_numpy(s).float()).shape, torch.from_numpy(mouse_act).float().shape)
            loss = F.smooth_l1_loss(q.brain_act(torch.from_numpy(s).float()), torch.from_numpy(mouse_act).float())
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    # optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(500):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        # while not done:
        for _ in range(1000):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            scores.append(score)
            score = 0.0
    env.close()
    return scores

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    baselines = []
    mouses = []
    for _ in range(100):
        baseline = main(use_mouse_act=False)
        mouse = main(use_mouse_act=True)
        baselines.append(np.asarray(baseline))
        mouses.append(np.asarray(mouse))
    baselines = np.asarray(baselines)
    mouses = np.asarray(mouses)
    plt.plot(baselines.mean(axis=0), color='blue', label='baseline')
    plt.plot(mouses.mean(axis=0), color='red', label='mouse')
    plt.legend()
    plt.savefig('mouse_plot.png')
