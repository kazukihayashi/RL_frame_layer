# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import chainer
import chainer.links as L
import chainer.functions as F
import copy
import myenv
import plotter
import csv
import SA
import time
from collections import deque
from matplotlib import animation

BATCH_SIZE = 100 # mini-batch size
CAPACITY = 1000 # maximum memory for Experience replay
GAMMA = 0.99 # learning rate
NUM_EPISODES = 1000
RECORD_INTERVAL = 5 # interval of recording total reward history

N_SAMPLE = 10 # number of samples to define initial temperature
N_ITERATIONS = 1000 # number of iterations for one optimization
#N_NEIGHBORS = 1 # number of neighbor solutions of SA
COOLING_INTERVAL = 20 # cooling interval in SA

class NN(chainer.Chain):
    def __init__(self,n_states,n_actions):
        super(NN,self).__init__(
            l1 = L.Linear(n_states,100),
            l2 = L.Linear(100,100),
            l3 = L.Linear(100,100),
            l4 = L.Linear(100,n_actions)
        )
        
    def forward(self,x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

class Brain():
    def __init__(self,n_states,n_actions):
        self.n_actions = n_actions
        self.model = NN(n_states,n_actions)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)
        
        self.memory = deque()
        self.loss = 0.0
        self.step = 0
        
    def store_experience(self,state,action,reward,state_next,ep_end):
        self.memory.append((state,action,reward,state_next,ep_end))
        if len(self.memory) > CAPACITY:
            self.memory.popleft() # eliminate the first element
            
    def shuffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)
    
    def parse_batch(self,batch):
        state,action,reward,state_next,ep_end = [],[],[],[],[]
        for i in range(BATCH_SIZE):
            state.append(batch[i][0])
            action.append(batch[i][1])
            reward.append(batch[i][2])
            state_next.append(batch[i][3])
            ep_end.append(batch[i][4])
        state = np.array(state,dtype=np.float32)
        action = np.array(action,dtype=np.int8)
        reward = np.array(reward,dtype=np.float32)
        state_next = np.array(state_next,dtype=np.float32)
        ep_end = np.array(ep_end,dtype = np.bool)
        return state,action,reward,state_next,ep_end
    
    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        mem = self.shuffle_memory()
        batch = mem[0:BATCH_SIZE]
        state,action,reward,state_next,ep_end = self.parse_batch(batch)
        self.model.cleargrads()
        loss = self.calc_loss(state,action,reward,state_next,ep_end)
        loss.backward()
        self.optimizer.update()
    
    def calc_loss(self,state,action,reward,state_next,ep_end):
        s = chainer.Variable(state)
        s_next = chainer.Variable(state_next)
        Q_NN = self.model.forward(s)
        tmp = self.target_model.forward(s_next)
        tmp = list(map(np.max, tmp.data))
        max_Q_next = np.asanyarray(tmp,dtype=np.float32)
        Q_target = np.asanyarray(copy.deepcopy(Q_NN.data),dtype=np.float32)
        for i in range(BATCH_SIZE):
            Q_target[i,action[i]] = reward[i] + (GAMMA * max_Q_next[i]) * (not ep_end[i])
        loss = F.mean_squared_error(Q_NN,Q_target)
        return loss
    
    def decide_action(self,state,eps):
        
        s = chainer.Variable(state)

        if np.random.rand() > eps:
            Q = self.model.forward(s)
            Q = Q.data[0]
            a = np.argmax(Q)
        else:
            a = np.random.randint(0,self.n_actions)

        return a

    def n_greedy_actions(self,state,n):
        
        s = chainer.Variable(state)

        Q = self.model.forward(s)
        Q = Q.data[0]
        Q -= np.min(Q)
        a = np.random.choice(self.n_actions,n,p=Q/np.sum(Q))

        return a

class Agent():
    def __init__(self,n_states,n_actions):
        self.brain = Brain(n_states,n_actions)
        self.step = 0
        self.target_update_freq = 5
        
    def update_q_function(self):
        self.brain.experience_replay()
        if self.step % self.target_update_freq == 0:
            self.brain.target_model = copy.deepcopy(self.brain.model)
        self.step += 1
        
    def get_action(self,state,eps=0.2):
        action = self.brain.decide_action(state,eps=eps)
        return action

    def n_greedy_actions(self,state,n):
        actions = self.brain.n_greedy_actions(state,n=n)
        return actions
    
    def memorize(self,state,action,reward,state_next,ep_end):
        self.brain.store_experience(state,action,reward,state_next,ep_end)

class Environment():
    def __init__(self):
        self.env = gym.make('myenv-v0')
        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)

        obs = self.env.reset()
        print('initial observation:', obs)

        self.env._render(mode='shape',name='init_shape')
        self.env._render(mode='section',name='init_section')
        self.env._render(mode='ratio',name='init_ratio')

        action = self.env.action_space.sample()
        obs, r, done, info = self.env.step(action)
        print('next observation:', obs)
        print('reward:', r)
        print('done:', done)
        print('info:', info)

        self.n_states = 22
        self.n_actions = 5
        self.agent = Agent(self.n_states,self.n_actions)

        ###self.agent.brain.model.to_gpu(0)
        
        
    def Learn(self):

        history = np.zeros(NUM_EPISODES//RECORD_INTERVAL,dtype=float)
        top_score = 0.0
        top_scored_iteration = -1
        top_scored_model = []
        n_analysis_until_best = 0
        n_analysis = 0

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            total_reward = 0.0
            
            for t in range(self.env.MAX_STEPS):

                state1 = observation["sec"].astype(np.float32).reshape((1,6))
                state2 = observation["sec_bound"].astype(np.float32).reshape((1,4))
                state3 = observation["stress"].astype(np.float32).reshape((1,6))
                state4 = observation["cof"].astype(np.float32).reshape((1,4))
                state5 = observation["fl_bound"].astype(np.float32).reshape((1,2))
                state = np.concatenate((state1,state2,state3,state4,state5),axis=1)
                action = self.agent.get_action(state)
                observation, reward, ep_end, _ = self.env._step(action) # Since info is unncessary, use "_"
                state_next1 = observation["sec"].astype(np.float32).reshape((1,6))
                state_next2 = observation["sec_bound"].astype(np.float32).reshape((1,4))
                state_next3 = observation["stress"].astype(np.float32).reshape((1,6))
                state_next4 = observation["cof"].astype(np.float32).reshape((1,4))
                state_next5 = observation["fl_bound"].astype(np.float32).reshape((1,2))        
                state_next = np.concatenate((state_next1,state_next2,state_next3,state_next4,state_next5),axis=1)
                self.agent.memorize(state,action,reward,state_next,ep_end)
                self.agent.update_q_function()
                total_reward += reward
                if ep_end:
                    break
                            
            print("episode {0}: step={1}, reward={2}".format(episode,t+1,total_reward))
            n_analysis += (self.env.steps+1) * 2 + 1

            if episode % RECORD_INTERVAL == 0:
                observation = self.env.reset()
                total_reward = 0.0
                for t in range(self.env.MAX_STEPS):
                    state1 = observation["sec"].astype(np.float32).reshape((1,6))
                    state2 = observation["sec_bound"].astype(np.float32).reshape((1,4))
                    state3 = observation["stress"].astype(np.float32).reshape((1,6))
                    state4 = observation["cof"].astype(np.float32).reshape((1,4))
                    state5 = observation["fl_bound"].astype(np.float32).reshape((1,2))
                    state = np.concatenate((state1,state2,state3,state4,state5),axis=1)
                    action = self.agent.get_action(state,eps=0.0)
                    observation, reward, ep_end, _ = self.env.step(action)
                    total_reward += reward
                    if ep_end:
                        break

                if(total_reward >= top_score):
                    top_score = total_reward
                    top_scored_iteration = episode
                    top_scored_model = copy.deepcopy(self.agent.brain.model)
                    n_analysis_until_best = n_analysis

                history[episode//RECORD_INTERVAL] = total_reward

        with open("result/info.txt", 'w') as f:
            f.write(str.format("total number of analysis:{0} \n",n_analysis))
            f.write(str.format("total number of analysis until best:{0} \n",n_analysis_until_best))
            f.write(str.format("top-scored iteration: {0} \n",top_scored_iteration+1))            

        plotter.graph(history,name="learn_history")
        with open("result/reward.csv", 'w') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            writer.writerow(history)

        np.savetxt('w1.dat',top_scored_model.l1.W.data)
        np.savetxt('w2.dat',top_scored_model.l2.W.data)
        np.savetxt('w3.dat',top_scored_model.l3.W.data)
        np.savetxt('w4.dat',top_scored_model.l4.W.data)
        np.savetxt('b1.dat',top_scored_model.l1.b.data)
        np.savetxt('b2.dat',top_scored_model.l2.b.data)
        np.savetxt('b3.dat',top_scored_model.l3.b.data)
        np.savetxt('b4.dat',top_scored_model.l4.b.data)
        self.agent.brain.model = copy.deepcopy(top_scored_model)

    def Illustrate_History(self):
        observation = self.env.reset()
        total_reward = 0.0
        for t in range(self.env.MAX_STEPS):
            state1 = observation["sec"].astype(np.float32).reshape((1,6))
            state2 = observation["sec_bound"].astype(np.float32).reshape((1,4))
            state3 = observation["stress"].astype(np.float32).reshape((1,6))
            state4 = observation["cof"].astype(np.float32).reshape((1,4))
            state5 = observation["fl_bound"].astype(np.float32).reshape((1,2))
            state = np.concatenate((state1,state2,state3,state4,state5),axis=1)
            action = self.agent.get_action(state,eps=0.0)
            observation, reward, ep_end, _ = self.env.step(action)
            total_reward += reward
            self.env._render(mode='section',name="{0}".format(self.env.steps))
            #self.env._render(mode='section',name="{0}_{1:.1f}".format(self.env.steps,reward))
            if ep_end:
                break
        self.env._render(mode='ratio',name="{0}_{1}".format(self.env.steps,'con'))
        print("total volume is", self.env.volume.value)
        print("total reward is", total_reward)

    def Read_Trained_NN(self):
        self.agent.brain.model.l1.W.data = np.loadtxt('w1.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l2.W.data = np.loadtxt('w2.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l3.W.data = np.loadtxt('w3.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l4.W.data = np.loadtxt('w4.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l1.b.data = np.loadtxt('b1.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l2.b.data = np.loadtxt('b2.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l3.b.data = np.loadtxt('b3.dat',dtype=np.float32, delimiter=' ')
        self.agent.brain.model.l4.b.data = np.loadtxt('b4.dat',dtype=np.float32, delimiter=' ')
    
    def Optimize(self):

        np.random.seed(0)

        all_history = []
        min_history = []
        samples = np.empty((N_SAMPLE,2))
        
        _ = self.env.reset()
        for i in range(N_SAMPLE):
            samples[i,0] = self.env.volume.value
        for i in range(N_SAMPLE):
            _ = self.env.reset()
            _, _ = self.env._step_for_optimization(np.random.randint(0,self.n_actions))
            samples[i,1] = self.env.volume.value

        _ = self.env.reset()
        init_x = [self.env.sec_num,self.env.section]
        init_f = self.env._step_for_optimization(self.n_actions-1)[1]
        optimizer = SA.SA(init_X=init_x,init_F=init_f,samples=samples)

        t1 = time.process_time()
        for cycle in range(10):

            optimizer.Reset()
            _ = self.env.reset()

            for i in range(N_ITERATIONS):
                state1 = self.env.state["sec"].astype(np.float32).reshape((1,6))
                state2 = self.env.state["sec_bound"].astype(np.float32).reshape((1,4))
                state3 = self.env.state["stress"].astype(np.float32).reshape((1,6))
                state4 = self.env.state["cof"].astype(np.float32).reshape((1,4))
                state5 = self.env.state["fl_bound"].astype(np.float32).reshape((1,2))
                state = np.concatenate((state1,state2,state3,state4,state5),axis=1)
                action = self.agent.get_action(state,eps=0.2)
                # action = np.random.randint(0,5)
                X, F = self.env._step_for_optimization(action)
                self.env.sec_num, self.env.section = copy.deepcopy(optimizer.Select(X, F)) # deepcopy is VITAL
                self.env.current_floor += 1
                if(self.env.current_floor == self.env.n_story):
                    self.env.current_floor -= self.env.n_story
                self.env.state_update()

                if(i % COOLING_INTERVAL == 0):
                    optimizer.Update_Temp()
                
            # make graph of history
            plotter.graph(optimizer.history,name=cycle)
            # append history
            all_history.append(optimizer.history)
            min_history.append(np.min(optimizer.history))

            # pick up the best solution
            self.env.sec_num, self.env.section = optimizer.best_X
            print("Best iteration is",optimizer.best_iter)
            self.env._render(mode='shape',name='opt{0:0=2}_shape'.format(cycle))
            self.env._render(mode='section',name='opt{0:0=2}_section'.format(cycle))
            self.env._render(mode='ratio',name='opt{0:0=2}_ratio'.format(cycle))

        t2 = time.process_time()
        print("process time:{:} seconds".format(t2-t1))

        with open('opt_history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_history)
        print("Maximum objfun is",np.max(min_history))    
        print("Median of objfun is",np.median(min_history))
        print("Minimum objfun is",np.min(min_history))
        print("Average objfun is",np.mean(min_history))
        print("Std.dev. objfun is",np.std(min_history))
                    
ramen_env = Environment()
#ramen_env.Learn()
ramen_env.Read_Trained_NN()
# ramen_env.Illustrate_History()
ramen_env.Optimize()

del ramen_env.env
