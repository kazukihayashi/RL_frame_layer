import sys
import gym
import numpy as np
import gym.spaces
import copy
import plotter
from ctypes import *

class RamenPlanning(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

	MAX_STEPS = 100

	np.random.seed(0)
	n_story = 8
	n_span = 3

	if(n_span < 2):
		raise ValueError("n_span and n_story must be larger than 1.")
	if(n_story < 4):
		raise ValueError("n_story must be larger than 3.")

	nk = (n_span+1)*(n_story+1)
	nm = (n_span+1)*n_story + n_span*n_story  # column + beam
	height = np.ones(n_story,dtype=float)*4.0
	#height[5] = 10.0
	span = 10.0
	total_reward = 0

	# node

	node = np.zeros((nk,3),dtype=np.float64,order='F')

	for i in range(n_story+1):
		node[i*(n_span+1):(i+1)*(n_span+1),1] = np.sum(height[0:i])
		for j in range(n_span+1):
			node[i*(n_span+1)+j,0] = span * j

	# member and its weak axis

	connectivity = []
	weak_axis = []

	for i in range(n_story):
		for j in range(n_span+1): # column of layer i
			connectivity.append([(n_span+1)*i+j ,(n_span+1)*(i+1)+j])
			weak_axis.append([1.0,0.0,0.0])
		for j in range(n_span): # beam of layer i
			connectivity.append([(n_span+1)*(i+1)+j,(n_span+1)*(i+1)+j+1])
			weak_axis.append([0.0,1.0,0.0])

	connectivity = np.asarray(connectivity,dtype=np.int64,order='F')+1
	weak_axis = np.asarray(weak_axis,dtype=np.float64,order='F')

	# section

	section = np.zeros((n_story*(2*n_span+1),6),dtype=np.float64,order='F')

	column_section_list ={
		## A, Ix, Iz(strong), Iy(weak), Zz(strong), Zy(weak) [metric]
		200:(85.3/1.0E4, 2*4860/1.0E8, 4860/1.0E8, 4860/1.0E8, 486/1.0E6, 486/1.0E6), # 200x200x12 cold-rolled
		250:(109.3/1.0E4, 2*10100/1.0E8, 10100/1.0E8, 10100/1.0E8, 805/1.0E6, 805/1.0E6), # 250x250x12 cold-rolled
		300:(173.0/1.0E4, 2*22600/1.0E8, 22600/1.0E8, 22600/1.0E8, 1510/1.0E6, 1510/1.0E6), # 300x300x16 cold-rolled
		350:(239.2/1.0E4, 2*42400/1.0E8, 42400/1.0E8, 42400/1.0E8, 2420/1.0E6, 2420/1.0E6), # 350x350x19 cold-rolled
		400:(307.7/1.0E4, 2*69500/1.0E8, 69500/1.0E8, 69500/1.0E8, 3480/1.0E6, 3480/1.0E6), # 400x400x22 cold-pressed
		450:(351.7/1.0E4, 2*103000/1.0E8, 103000/1.0E8, 103000/1.0E8, 4560/1.0E6, 4560/1.0E6), # 450x450x22
		500:(442.8/1.0E4, 2*159000/1.0E8, 159000/1.0E8, 159000/1.0E8, 6360/1.0E6, 6360/1.0E6), # 500x500x25
		550:(492.8/1.0E4, 2*217000/1.0E8, 217000/1.0E8, 217000/1.0E8, 7900/1.0E6, 7900/1.0E6), # 550x550x25
		600:(542.8/1.0E4, 2*288000/1.0E8, 288000/1.0E8, 288000/1.0E8, 9620/1.0E6, 9620/1.0E6), # 600x600x25
		650:(656.3/1.0E4, 2*407000/1.0E8, 407000/1.0E8, 407000/1.0E8, 12500/1.0E6, 12500/1.0E6), # 650x650x28
		700:(712.3/1.0E4, 2*518000/1.0E8, 518000/1.0E8, 518000/1.0E8, 14800/1.0E6, 14800/1.0E6), # 700x700x28
		750:(866.3/1.0E4, 2*717000/1.0E8, 717000/1.0E8, 717000/1.0E8, 19100/1.0E6, 19100/1.0E6), # 750x750x32
		800:(930.3/1.0E4, 2*884000/1.0E8, 884000/1.0E8, 884000/1.0E8, 22100/1.0E6, 22100/1.0E6), # 800x800x32
		850:(994.3/1.0E4, 2*1070000/1.0E8, 1070000/1.0E8, 1070000/1.0E8, 25300/1.0E6, 25300/1.0E6) # 850x850x32
		}
	column_plastic_section_modulus = {200:588/1.0E6, 250:959/1.0E6, 300:1810/1.0E6, 350:2910/1.0E6, 400:4220/1.0E6, 450:5490/1.0E6, 500:7660/1.0E6, 550:9460/1.0E6, 600:11400/1.0E6, 650:14900/1.0E6, 700:17600/1.0E6, 750:22800/1.0E6, 800:26200/1.0E6, 850:29900/1.0E6}

	beam_section_list ={
		## A, Ix, Iz(strong), Iy(weak), Zz(strong), Zy(weak) [metric]
		200:(38.11/1.0E4, 2*2630/1.0E8, 2630/1.0E8, 507/1.0E8, 271/1.0E6, 67.6/1.0E6), # 194x150x6x9
		250:(55.49/1.0E4, 2*6040/1.0E8, 6040/1.0E8, 984/1.0E8, 495/1.0E6, 112/1.0E6), # 244x175x7x11
		300:(71.05/1.0E4, 2*11100/1.0E8, 11100/1.0E8, 1600/1.0E8, 756/1.0E6, 160/1.0E6), # 294x200x8x12
		350:(99.53/1.0E4, 2*21200/1.0E8, 21200/1.0E8, 3650/1.0E8, 1250/1.0E6, 292/1.0E6), # 340x250x9x14 middle H
		400:(110.0/1.0E4, 2*31600/1.0E8, 31600/1.0E8, 2540/1.0E8, 1580/1.0E6, 254/1.0E6), # 400x200x9x19 super high-slend H (SHH)
		450:(126.0/1.0E4, 2*45900/1.0E8, 45900/1.0E8, 2940/1.0E8, 2040/1.0E6, 294/1.0E6), # 450x200x9x22
		500:(152.5/1.0E4, 2*70700/1.0E8, 70700/1.0E8, 5730/1.0E8, 2830/1.0E6, 459/1.0E6), # 500x250x9x22
		550:(157.0/1.0E4, 2*87300/1.0E8, 87300/1.0E8, 5730/1.0E8, 3180/1.0E6, 459/1.0E6), # 550x250x9x22
		600:(192.5/1.0E4, 2*121000/1.0E8, 121000/1.0E8, 6520/1.0E8, 4040/1.0E6, 522/1.0E6), # 600x250x12x25
		650:(198.5/1.0E4, 2*145000/1.0E8, 145000/1.0E8, 6520/1.0E8, 4460/1.0E6, 522/1.0E6), # 650x250x12x25
		700:(205.8/1.0E4, 2*173000/1.0E8, 173000/1.0E8, 6520/1.0E8, 4940/1.0E6, 522/1.0E6), # 700x250x12x25
		750:(267.9/1.0E4, 2*261000/1.0E8, 261000/1.0E8, 12600/1.0E8, 6970/1.0E6, 841/1.0E6), # 750x300x14x28
		800:(274.9/1.0E4, 2*302000/1.0E8, 302000/1.0E8, 12600/1.0E8, 7560/1.0E6, 841/1.0E6), # 800x300x14x28
		850:(297.8/1.0E4, 2*355000/1.0E8, 355000/1.0E8, 12600/1.0E8, 8350/1.0E6, 842/1.0E6) # 850x300x16x28
	}
	beam_plastic_section_modulus = {200:301/1.0E6, 250:550/1.0E6, 300:842/1.0E6, 350:1380/1.0E6, 400:1770/1.0E6, 450:2750/1.0E6, 500:3130/1.0E6, 550:3520/1.0E6, 600:4540/1.0E6, 650:5030/1.0E6, 700:5580/1.0E6, 750:7850/1.0E6, 800:8520/1.0E6, 850:9540/1.0E6}

	# material

	DESIGN_STRENGTH_BEAM = 235e6 #[N/m^2]
	DESIGN_STRENGTH_COLUMN = 295e6 #[N/m^2]
	E = 2.05E11
	G = 7.9E10
	material = np.ones((nm,2),dtype=np.float64,order='F')
	material[:,0] *= E
	material[:,1] *= G

	# support condition

	support_array = []
	for i in range(n_span+1):
		support_array.append([1,1,1,1,1,1])
	for i in range(nk-(n_span+1)):
		support_array.append([0,0,1,1,1,0])
	support = np.asarray(support_array,dtype=np.int64,order='F')

	# call dll

	my_mod = CDLL("ramen_analysis.dll")
	my_mod.ramen_linear_analysis.argtypes = [
		### input ###
		np.ctypeslib.ndpointer(dtype = np.float64), # node[nk,3] 0:x,1:y,2:z
		np.ctypeslib.ndpointer(dtype = np.int64),   # connectivity[nm,2] 0:start,1:end
		np.ctypeslib.ndpointer(dtype = np.float64), # section[nm,6] 0:A,1:Ix(torsion),2:Iz(strong),3:Iy(weak),4:Zz(strong),5:Zy(weak)
		np.ctypeslib.ndpointer(dtype = np.float64), # weak_axis[nm,3] 0:x,1:y,2:z
		np.ctypeslib.ndpointer(dtype = np.float64), # material[nm,2] 0:E,1:G
		np.ctypeslib.ndpointer(dtype = np.int64),   # support[nk,6] 0:x,1:y,2:z,3:xr,4:yr,5:zr(0:free,1:fix)
		np.ctypeslib.ndpointer(dtype = np.float64), # load[nk,6] 0:x,1:y,2:z,3:xr,4:yr,5:zr
		POINTER(c_int64),                             # nk (number of nodes)
		POINTER(c_int64),                             # nm (number of members)
		### output ###
		np.ctypeslib.ndpointer(dtype = np.float64), # displacement[nk,6] 0:x,1:y,2:z,3:xr,4:yr,5:zr
		np.ctypeslib.ndpointer(dtype = np.float64), # stress[nm,5] 0:axial,1:bending(start,weak),2:bending(start,strong),3:bending(end,weak),4:bending(end,strong)
		POINTER(c_double),                          # total_structural_volume
		POINTER(c_double)                           # compliance
		]
	my_mod.ramen_linear_analysis.restype = c_void_p

	disp = np.empty((nk,6),dtype=np.float64,order='F')
	stress = np.empty((nm,5),dtype=np.float64,order='F')
	volume = c_double(0.0)
	compliance = c_double(0.0)
	
	#min_volume = column_section_list[200][0]*(n_span+1)*n_story + beam_section_list[200][0]*(n_span)*n_story

	def __init__(self):
		super().__init__()
		# Setting of action_space, observation_space and reward_range
		self.action_space = gym.spaces.Discrete(5)  # column/beam, upsize/downsize & keep
		self.observation_space = gym.spaces.Dict({
			"sec":gym.spaces.Box(low=0.0,high=1.0,shape=(2,3),dtype=float),
			"sec_bound":gym.spaces.Box(low=0.0,high=1.0,shape=(2,2),dtype=int), # column/beam, lower/upper bound
			"stress":gym.spaces.Box(low=0.0,high=10.0,shape=(2,3),dtype=float), # column/beam, -1/0/+1 layer
			"cof":gym.spaces.Box(low=0.0,high=50.0,shape=(4,),dtype=float), # from lower to upper layer
			"fl_bound":gym.spaces.Box(low=0.0,high=1.0,shape=(2,),dtype=int), # 1st/top floor
		})
		
		self.reward_range = [0.0, 1.0]
		self._reset()

	def _reset(self,variable=None):
		# initialize variables
		self.state = { # (0:column, 1:beam)
		"sec":np.zeros((2,3),dtype=float),
		"sec_bound":np.zeros((2,2),dtype=int),
		"stress":np.empty((2,3),dtype=float),
		"cof":np.empty(4,dtype=float),
		"fl_bound":np.zeros(2,dtype=int)
		}
		self.current_floor = 0
		self.done = False
		self.steps = 0

		if variable is None:
			init_sec_num = 400
			self.sec_num = np.ones((2*self.n_span+1)*self.n_story,dtype=int)*init_sec_num
			for i in range(self.n_story):
				for j in range(self.n_span+1): # column of layer i
					self.section[i*(2*self.n_span+1)+j] = self.column_section_list[init_sec_num]
				for j in range(self.n_span): # beam of layer i
					self.section[i*(2*self.n_span+1)+j+self.n_span+1] = self.beam_section_list[init_sec_num]
		else:
			self.set_sec_from_variable(variable)

		self.state_update()
		self.total_reward = 0.0

		return self.state

	def set_sec_from_variable(self,variable):
		# variable [0,1]
		v = 200+np.round(variable*9)*50
		for i in range(self.n_story):
			for j in range(self.n_span+1): # column of layer i
				self.sec_num[i*(2*self.n_span+1)+j] = v[2*i]
				self.section[i*(2*self.n_span+1)+j] = self.column_section_list[v[2*i]]
			for j in range(self.n_span): # beam of layer i
				self.sec_num[i*(2*self.n_span+1)+j+self.n_span+1] = v[2*i+1]
				self.section[i*(2*self.n_span+1)+j+self.n_span+1] = self.beam_section_list[v[2*i+1]]
		return

	def update_load(self):

		# static earthquake load

		total_weight_for_earthquake = 0.0 #[N]
		alpha_i_for_earthquake = np.zeros(self.n_story) # ratio of upper mass to total mass
		layerwise_weight_for_frame = np.zeros(self.n_story)

		for i in range(self.n_story-1,-1,-1):
			volume_layer_i = self.section[i*(2*self.n_span+1),0]*self.height[i]*(self.n_span+1) + self.section[i*(2*self.n_span+1)+self.n_span+1,0]*self.span*self.n_span
			structural_weight_layer_i = volume_layer_i * 77000 #[N/m^3]
			layerwise_weight_for_frame[self.n_story-1-i] = structural_weight_layer_i +  10000 * self.span * self.n_span # live load(for frame, office)
			total_weight_for_earthquake += structural_weight_layer_i +  5000 * self.span * self.n_span # live load(for earthquake, office)
			alpha_i_for_earthquake[i] = total_weight_for_earthquake
		alpha_i_for_earthquake /= total_weight_for_earthquake
		qi = 0.3 * total_weight_for_earthquake * np.sqrt(alpha_i_for_earthquake)

		pi = np.array(qi)
		for i in range(self.n_story-1):
			pi[i] -= pi[i+1]

		# loading condition

		self.load = np.zeros((self.nk,6),dtype=np.float64,order='F')

		for i in range(self.n_story):
			for j in range(self.n_span+1):
				if j == 0 or j == self.n_span:
					self.load[(self.n_span+1)*(i+1)+j][0] = pi[i]*0.5/self.n_span
					self.load[(self.n_span+1)*(i+1)+j][1] = -layerwise_weight_for_frame[i]*0.5/self.n_span
				else:
					self.load[(self.n_span+1)*(i+1)+j][0] = pi[i]/self.n_span
					self.load[(self.n_span+1)*(i+1)+j][1] = -layerwise_weight_for_frame[i]/self.n_span

	def state_update(self, SectionChangedAndDoAnalyze=True):

		if SectionChangedAndDoAnalyze:
			# update loading condition
			self.update_load()
			# conduct structural analysis to obtain displacement, stress, total structural volume, and compliance
			self.my_mod.ramen_linear_analysis(self.node,self.connectivity,self.section,self.weak_axis,self.material,self.support,self.load,c_int64(self.nk),c_int64(self.nm),self.disp,self.stress,self.volume,self.compliance)

		# section to state

		self.state["sec"][0,0] = self.sec_num[(self.n_span*2+1)*(self.current_floor-1)]/650 if self.current_floor != 0 else 0
		self.state["sec"][1,0] = self.sec_num[(self.n_span*2+1)*(self.current_floor-1)+self.n_span+1]/650 if self.current_floor != 0 else 0
		self.state["sec"][0,1] = self.sec_num[(self.n_span*2+1)*self.current_floor]/650
		self.state["sec"][1,1] = self.sec_num[(self.n_span*2+1)*self.current_floor+self.n_span+1]/650
		self.state["sec"][0,2] = self.sec_num[(self.n_span*2+1)*(self.current_floor+1)]/650 if self.current_floor != self.n_story-1 else 0
		self.state["sec"][1,2] = self.sec_num[(self.n_span*2+1)*(self.current_floor+1)+self.n_span+1]/650 if self.current_floor != self.n_story-1 else 0

		# section_bound to state

		self.state["sec_bound"][0,0] = 1 if self.sec_num[(self.n_span*2+1)*self.current_floor] == 200 else 0 # column, lower bound
		self.state["sec_bound"][0,1] = 1 if self.sec_num[(self.n_span*2+1)*self.current_floor] == 650 else 0 # column, upper bound
		self.state["sec_bound"][1,0] = 1 if self.sec_num[(self.n_span*2+1)*self.current_floor+self.n_span+1] == 200 else 0 # beam, lower bound
		self.state["sec_bound"][1,1] = 1 if self.sec_num[(self.n_span*2+1)*self.current_floor+self.n_span+1] == 650 else 0 # beam, upper bound

		# stress to state

		stress_ratio = np.abs(self.stress/self.allowable_stress(self.stress))
		self.max_ratio = stress_ratio[:,0] + np.max(stress_ratio[:,1:5],axis=1)

		if(self.current_floor != 0):
			self.state["stress"][0,0] = np.max(self.max_ratio[(2*self.n_span+1)*(self.current_floor-1):(2*self.n_span+1)*(self.current_floor-1)+self.n_span+1]) # column of layer i-1
			self.state["stress"][1,0] = np.max(self.max_ratio[(2*self.n_span+1)*(self.current_floor-1)+self.n_span+1:(2*self.n_span+1)*self.current_floor]) # beam of layer i-1
		else:
			self.state["stress"][:,0] = 0.0

		self.state["stress"][0,1] = np.max(self.max_ratio[(2*self.n_span+1)*self.current_floor:(2*self.n_span+1)*self.current_floor+self.n_span+1]) # column of layer i
		self.state["stress"][1,1] = np.max(self.max_ratio[(2*self.n_span+1)*self.current_floor+self.n_span+1:(2*self.n_span+1)*(self.current_floor+1)]) # beam of layer i

		if(self.current_floor != self.n_story-1):
			self.state["stress"][0,2] = np.max(self.max_ratio[(2*self.n_span+1)*(self.current_floor+1):(2*self.n_span+1)*(self.current_floor+1)+self.n_span+1]) # column of layer i+1
			self.state["stress"][1,2] = np.max(self.max_ratio[(2*self.n_span+1)*(self.current_floor+1)+self.n_span+1:(2*self.n_span+1)*(self.current_floor+2)]) # beam of layer i+1
		else:
			self.state["stress"][:,2] = 0.0

		# column-to-beam strength ratio to state

		if(self.current_floor <= 1):
			self.state["cof"][0] = 0.0
			if(self.current_floor == 0):
				self.state["cof"][1] = 0.0
			else:
				self.state["cof"][1] = np.min([self.column_to_beam_strength_ratio(self.current_floor-1,j) for j in range(self.n_span)])
		else:
			self.state["cof"][0] = np.min([self.column_to_beam_strength_ratio(self.current_floor-2,j) for j in range(self.n_span)])
			self.state["cof"][1] = np.min([self.column_to_beam_strength_ratio(self.current_floor-1,j) for j in range(self.n_span)])

		self.state["cof"][2] = np.min([self.column_to_beam_strength_ratio(self.current_floor,j) for j in range(self.n_span)])

		if(self.current_floor == self.n_story-1):
			self.state["cof"][3] = 0.0
		else:
			self.state["cof"][3] = np.min([self.column_to_beam_strength_ratio(self.current_floor+1,j) for j in range(self.n_span)])

		# floor to state

		self.state["fl_bound"][0] = 1 if self.current_floor == 0 else 0
		self.state["fl_bound"][1] = 1 if self.current_floor == self.n_story-1 else 0

		return

	def allowable_stress(self,stress):

		critical_slenderness_ratio = np.sqrt(np.power(np.pi,2)*self.E/(0.6*self.DESIGN_STRENGTH_COLUMN))
		allowable_stress = np.empty((self.nm,5),dtype=float)
		
		for i in range(self.n_story):

			# column
			for j in range(self.n_span+1):
				if stress[(self.n_span*2+1)*i+j,0] < 0: # compression
					slenderness_ratio = self.height[i]*0.65/np.sqrt(self.section[(self.n_span*2+1)*i+j,3]/self.section[(self.n_span*2+1)*i+j,0])
					a = slenderness_ratio/critical_slenderness_ratio
					if(a < 1):
						allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_COLUMN*(1.0-0.4*np.power(a,2))/(3.0/2.0+np.power(a,2)*2.0/3.0)
					else:
						allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_COLUMN*18.0/(65.0*np.power(a,2))
				else: # tension
					allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_COLUMN/1.5

			allowable_stress[(self.n_span*2+1)*i:(self.n_span*2+1)*i+(self.n_span+1),1:5] = self.DESIGN_STRENGTH_COLUMN/1.5 # bending
			
			# beam
			for j in range(self.n_span+1,self.n_span*2+1):
				if stress[(self.n_span*2+1)*i+j,0] < 0: # compression
					slenderness_ratio = self.height[i]*0.65/np.sqrt(self.section[(self.n_span*2+1)*i+j,3]/self.section[(self.n_span*2+1)*i+j,0])
					a = slenderness_ratio/critical_slenderness_ratio
					if(a < 1):
						allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_BEAM*(1.0-0.4*np.power(a,2))/(3.0/2.0+np.power(a,2)*2.0/3.0)
					else:
						allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_BEAM*18.0/(65.0*np.power(a,2))
				else: # tension
					allowable_stress[(self.n_span*2+1)*i+j,0] = self.DESIGN_STRENGTH_BEAM/1.5

			allowable_stress[(self.n_span*2+1)*i+(self.n_span+1):(self.n_span*2+1)*(i+1),1:5] = self.DESIGN_STRENGTH_BEAM/1.5 # bending

		return allowable_stress*1.5 # short-term(*1.5), long-term(*1.0)

	def column_to_beam_strength_ratio(self, i, j):
		'''
		(input)
		i<int>: layer
		j<int>: j th node of i th layer
		(output)
		ratio<float>: column-to-beam strength ratio of j th node of i th layer
		'''

		# bottom column
		axial_force_ratio = self.stress[(2*self.n_span+1)*i+j,0]/self.DESIGN_STRENGTH_COLUMN
		if(axial_force_ratio < 0.5):
			alpha = 1-4*np.power(axial_force_ratio,2)/3
		else:
			alpha = 4*(1-axial_force_ratio)/3
		Mpc_bottom = alpha*self.DESIGN_STRENGTH_COLUMN*self.column_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*i]] 

		# upper column
		if (i!=self.n_story-1):
			axial_force_ratio = self.stress[(2*self.n_span+1)*(i+1)+j,0]/self.DESIGN_STRENGTH_COLUMN
			if(axial_force_ratio < 0.5):
				alpha = 1-4*np.power(axial_force_ratio,2)/3
			else:
				alpha = 4*(1-axial_force_ratio)/3
			Mpc_upper = alpha*self.DESIGN_STRENGTH_COLUMN*self.column_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*(i+1)]] 
		else:
			Mpc_upper = 0.0

		# left and right beams
		if(j%(self.n_span+1)==0):
			Mpb_left = 0.0
			Mpb_right = self.DESIGN_STRENGTH_BEAM*self.beam_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*i+j+self.n_span+1]]
		elif(j%(self.n_span+1)==self.n_span):
			Mpb_left = self.DESIGN_STRENGTH_BEAM*self.beam_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*i+j+self.n_span]]
			Mpb_right = 0.0
		else:
			Mpb_left = self.DESIGN_STRENGTH_BEAM*self.beam_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*i+j+self.n_span]]
			Mpb_right = self.DESIGN_STRENGTH_BEAM*self.beam_plastic_section_modulus[self.sec_num[(2*self.n_span+1)*i+j+self.n_span+1]]

		ratio = (Mpc_bottom+Mpc_upper)/(Mpb_left+Mpb_right)		

		return ratio


	def _step(self, action):

		'''
		action(int):
		0: column up
		1: beam up
		2: column down
		3: beam down
		4: keep
		'''

		# proceed to next step

		i_updown = action // 2 # 0:up 1:down 2:keep
		i_type = action % 2 # 0:column 1:beam

		updown = {0:-50,1:50,2:0}
		observation_value = self.sec_num[(self.n_span*2+1)*self.current_floor+(self.n_span+1)*i_type] + updown[i_updown]

		# confirm if satisfy constraints
		
		ok = True
		if (200 <= observation_value <= 650):

			self.sec_num[(self.n_span*2+1)*self.current_floor+(self.n_span+1)*i_type:(self.n_span*2+1)*self.current_floor+self.n_span+1+self.n_span*i_type] = observation_value
			if(i_type == 0):
				self.section[(self.n_span*2+1)*self.current_floor:(self.n_span*2+1)*self.current_floor+self.n_span+1] = self.column_section_list[observation_value]
			else:
				self.section[(self.n_span*2+1)*self.current_floor+self.n_span+1:(self.n_span*2+1)*(self.current_floor+1)] = self.beam_section_list[observation_value]
			self.state_update()

			if np.all(self.state["stress"]<0.8):

				if self.current_floor == self.n_story-1:
					if np.any(self.state["cof"][0:2]<1.5):
						ok = False
				elif self.current_floor == self.n_story-2:
					if np.any(self.state["cof"][0:3]<1.5):
						ok = False
				elif self.current_floor == 1:
					if np.any(self.state["cof"][1:4]<1.5):
						ok = False
				elif self.current_floor == 0:
					if np.any(self.state["cof"][2:4]<1.5):
						ok = False
				else:
					if np.any(self.state["cof"]<1.5):
						ok = False

			else:
				ok = False
		else:
			ok = False
		
		if ok:
			reward = (200+200)/(self.sec_num[(self.n_span*2+1)*self.current_floor]+self.sec_num[(self.n_span*2+1)*self.current_floor+(self.n_span+1)])
		else:
			reward = 0.0

		self.total_reward += reward
		self.steps += 1
		self.current_floor += 1
		if(self.current_floor == self.n_story):
			self.current_floor -= self.n_story
		self.state_update(SectionChangedAndDoAnalyze=False)

		self.done = self._is_done()

		return self.state, reward, self.done, {}

	def _step_for_optimization(self):

		self.state_update()

		objfun = self.volume.value

		# allowable stress constraint
		if any(self.max_ratio > 0.8):
			objfun = np.inf
			
		# column-to-beam strength ratio constraint
		for i in range(self.n_story-1):
			for j in range(self.n_span+1):
				if self.column_to_beam_strength_ratio(i,j)<1.5:
					objfun = np.inf
					break
			else:
				continue
			break
					
		return objfun

	def _render(self, name, mode='section', close=False):
		'''
		mode='shape'
		  No annotation is provided. Just nodes and lines are illustrated. 
		mode='section'
		  In addition to 'shape' illustration, section numbers are provided.
		mode='ratio'
		  In addition to 'shape' illustration, safety ratio of members and column-to-beam strength ratio of nodes are provided.
		'''
		# state update to retrieve safety ratio of members
		self.state_update(SectionChangedAndDoAnalyze=True)
		
		# member info
		l_width, l_color, l_text = [],[],[]

		for i in range(self.nm):
			l_width.append(self.sec_num[i])
			l_color.append("Red" if self.max_ratio[i]>1.0 else "yellow" if self.max_ratio[i]>0.8 else "Black")
			if mode == 'section':
				l_text.append('{0}'.format(self.sec_num[i]))
			elif mode == 'ratio':
				l_text.append('{:.2f}'.format(self.max_ratio[i]))

		# node info
		node2d = self.node[:,0:2]
		n_color, n_text = [], []
		for j in range(self.n_span+1):
			n_color.append("Black")
			n_text.append('{:.2f}'.format(0))
		for i in range(self.n_story-1):
			for j in range(self.n_span+1):
				cof = self.column_to_beam_strength_ratio(i,j)
				n_color.append("Orange" if cof<1.5 else "Green")
				n_text.append('{:.2f}'.format(cof))
		for j in range(self.n_span+1):
			n_color.append("Black")
			n_text.append('{:.2f}'.format(self.column_to_beam_strength_ratio(self.n_story-1,j)))
		
		if mode == 'shape':
			outfile = plotter.Draw(node2d,self.connectivity-1,node_color=n_color,line_width=l_width,line_color=l_color,node_text=None,line_text=None,name=name)
		elif mode == 'section':
			outfile = plotter.Draw(node2d,self.connectivity-1,node_color=n_color,line_width=l_width,line_color=l_color,node_text=None,line_text=l_text,name=name)
		elif mode == 'ratio':
			outfile = plotter.Draw(node2d,self.connectivity-1,node_color=n_color,line_width=l_width,line_color=l_color,node_text=n_text,line_text=l_text,name=name)
		else:
			raise Exception("Unexpected mode selection.")

		return outfile

	def _close(self):
		pass

	def _seed(self, seed=None):
		pass

	def _is_done(self):
		if self.steps > self.MAX_STEPS:
			return True
		else:
			return False

