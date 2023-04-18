import numpy as np
import copy
import matplotlib.pyplot as plt

class PSO():
    def __init__(self,init_X,init_F,bounds):
        self.init_X = init_X
        self.init_F = init_F
        self.nvar = np.size(bounds,0)
        self.bounds = bounds
        self.np = 5
        self.Reset()
        return

    def Reset(self):
        self.p_best_f = None
        self.f_best_f = np.Inf
        self.swarm = []
        self.history = []
        for _ in range(self.np):
            self.swarm.append(Particle(self.init_X,self.init_F,self.bounds))
        self.p_best_g = self.init_X
        self.f_best_g = self.init_F
        self.c_best_g = False # constriant function satisfaction
        return

class Particle():
    def __init__(self,init_X,init_F,bounds):
        self.nvar = np.size(bounds,0)
        self.bounds = bounds
        self.range = bounds[:,1] - bounds[:,0]

        self.p_i = init_X
        self.v_i = (np.random.rand(self.nvar)-0.5)*self.range*0.1
        self.p_best_i = init_X
        self.f_best_i = init_F

        self.C1 = 2.0
        self.C2 = 2.0
        return

    def Update(self, p_best_g, progress):
        v_cognitive = self.C1 * np.random.rand(self.nvar) * (self.p_best_i-self.p_i)
        v_social = self.C2 * np.random.rand(self.nvar) * (p_best_g-self.p_i)
        # v = (0.4+0.5*(1.0-progress)) * self.v_i + v_cognitive + v_social
        v = 0.7 * self.v_i + v_cognitive + v_social

        v[v>self.range*0.1] = 0.1*self.range[v>self.range*0.1]
        v[v<-self.range*0.1] = 0.1*-self.range[v<-self.range*0.1]

        p_i_before = copy.copy(self.p_i)
        self.p_i = self.p_i + v

        l = self.p_i < self.bounds[:,0]
        self.p_i[l] = self.bounds[l,0]
        u = self.p_i > self.bounds[:,1]
        self.p_i[u] = self.bounds[u,1]

        self.v_i = self.p_i - p_i_before
        return
