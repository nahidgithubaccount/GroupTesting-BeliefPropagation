import numpy as np
import random
from joblib import Parallel, delayed
import time

def Random_Regular_Sparse_Matrix(n,m,d_v):
	#np.random.seed(0)
	d_c = np.floor(n*d_v/m).astype(int)
	E = n*d_v
	Flag = True
	while Flag:
		I = np.random.permutation(E)+1
		H = np.zeros((m,n))
		i = 0
		while i<m:
			H[i][np.ceil(I[:d_c]/d_v).astype(int)-1] = 1
			if np.sum(H[i])!=d_c:
				if i!=m-1:
					H[i][:] = 0
					I = np.random.permutation(I)
				else:
					break
			else:
				I = I[d_c:]
				i += 1
				if i==m:
					Flag = False
	
	return H

def Random_Bernoulli_Matrix(n,m,k,nu):
	#np.random.seed(0)
	H = (np.random.rand(m,n)<=nu/k)
	H = H.astype(int)
	return H

def Boolean_Measurements(H,x):
	y = np.dot(H,x).astype(int)
	y = y>0
	y = y.astype(int)
	return y

def Neighbors(H):
	m, n = H.shape
	N_c = []
	for i in range(m):
		I = np.nonzero(H[i])[0]
		N_c.append(I)
		
	HT = np.transpose(H)
	N_v = []
	for j in range(n):
		I = np.nonzero(HT[j])[0]
		N_v.append(I)
	
	return N_c, N_v

def Binary_Symmetric_Channel(y,delta,trial):
	np.random.seed(trial)
	m = len(y)
	y_hat = [0 for i in range(m)]
	for i in range(m):
		if np.random.uniform()<=delta:
			y_hat[i] = (y[i]+1)%2
		else:
			y_hat[i] = y[i]
	return y_hat
	
def v_to_c_message(Temp_N_v,c,allc_to_v_pre_messages,q):
	Temp = [1-q,q]
	for i in range(len(Temp_N_v)):
		if Temp_N_v[i] != c:
			Temp[0] *= allc_to_v_pre_messages[i][0]
			Temp[1] *= allc_to_v_pre_messages[i][1]
	num = np.array(Temp)
	denum  = np.sum(num)
	messages = num/denum
	return messages

def c_to_v_messages(c_val,Temp_N_c,v,allv_to_c_pre_messages,delta):
	Temp = 1.0
	for i in range(len(Temp_N_c)):
		if Temp_N_c[i] != v:
			Temp *= allv_to_c_pre_messages[i][0]
			
	if c_val==0:
		num = np.array([delta+(1-2*delta)*Temp,delta])
	else:
		num = np.array([1-delta-(1-2*delta)*Temp,1-delta])
	
	denum = np.sum(num)
	messages = num/denum
	return messages

def Compute_LLR(n,q,N_v,allc_to_v_messages):
	LLR = np.array([0 for v in range(n)])
	for v in range(n):
		Temp_prob = np.array([1-q,q])
		for i in range(len(N_v[v])):
			Temp_prob[0] *= allc_to_v_messages[v][i][0]
			Temp_prob[1] *= allc_to_v_messages[v][i][1]
		sum_prob = np.sum(Temp_prob)
		Temp_prob = Temp_prob/sum_prob
		LLR[v] = np.log(Temp_prob[1]/Temp_prob[0])
	return LLR

def L1_norm(vec1,vec2):
	return np.sum(np.abs(vec1-vec2))
	
def L2_norm(vec1,vec2):
	return np.sum((vec1-vec2)**2)

def BP_Flooding(y_hat,n,m,k,N_c,N_v,q,delta,no_iter):
	for iter in range(no_iter):
		if iter==0:
			allc_to_v_messages = np.array([[[0.5,0.5] for i in range(len(N_v[v]))] for v in range(n)], dtype=object)
			allv_to_c_messages = np.array([[[1-q,q] for i in range(len(N_c[c]))] for c in range(m)], dtype=object)
			Pre_LLR = Compute_LLR(n,q,N_v,allc_to_v_messages)
		else:
			for c in range(m):
				c_val = y_hat[c]
				Temp_N_c = N_c[c]
				for v in Temp_N_c:
					Temp_mesages = c_to_v_messages(c_val,Temp_N_c,v,allv_to_c_messages[c],delta)
					c_index = np.where(N_v[v]==c)
					allc_to_v_messages[v][c_index[0][0]][0] = Temp_mesages[0]
					allc_to_v_messages[v][c_index[0][0]][1] = Temp_mesages[1]
			
			LLR = Compute_LLR(n,q,N_v,allc_to_v_messages)
			if L1_norm(LLR,Pre_LLR)<=0.001:
				break
			else:
				for i in range(n):
					Pre_LLR[i] = LLR[i]
				for v in range(n):
					Temp_N_v = N_v[v]
					for c in Temp_N_v:
						Temp_mesages = v_to_c_message(Temp_N_v,c,allc_to_v_messages[v],q)
						v_index = np.where(N_c[c]==v)
						allv_to_c_messages[c][v_index[0][0]][0] = Temp_mesages[0]
						allv_to_c_messages[c][v_index[0][0]][1] = Temp_mesages[1]
		
	x_hat1 = LLR>=0
	x_hat1 = x_hat1.astype(int)
	I_hat1 = np.where(x_hat1==1)[0]
	
	x_hat2 = np.array([0 for i in range(n)])
	sorted_LLR_indices = np.argsort(LLR)[::-1]
	x_hat2[sorted_LLR_indices[:k]] = 1
	I_hat2 = np.where(x_hat2==1)[0]
	
	return I_hat1, I_hat2
		
n = 150 # number of people in the population
m = 50 # number of tests
k = 5 # number of infected people 
delta = 0.05 # crossover probability of BSC channel
q = k/n # probability of an individual to be infected
no_iter = 10 # number of BP iterations
no_trials = 1024 # number of simulation trials
#d_v = 3 # number of 1's per column for regular sparse testing matrix
#H = Random_Regular_Sparse_Matrix(n,m,d_v)
nu = np.log(2) # nu/k is the probability that an entry in Bernoulli testing matrix is 1
H = Random_Bernoulli_Matrix(n,m,k,nu) # Bernoulli testing matrix

# N_c[c] = index set of variable nodes incident to check node c
# N_v[v] = index set of check nodes incident to variable node v
N_c, N_v = Neighbors(H)

def CodeRunner(trial):
	np.random.seed(trial)
	# x = a binary vector of length n, representing the status of individuals in the population
	x = np.array([0 for i in range(n)])
	# I = sorted index set of infected people
	I = np.random.choice(n,k,replace=False)
	I = np.sort(I) 
	for i in I:
		x[i] = 1
	
	# y = a binary vector of length m, representing the true test results
	y = Boolean_Measurements(H,x)
	# y_hat = a binary vector of length m, representing the noisy version of test results
	y_hat = Binary_Symmetric_Channel(y,delta,trial)
	
	# BP algorithm with flooding
	# I_hat1 = Estimate of I when k is unknown
	# I_hat2 = Estimate of I when k is known
	I_hat1, I_hat2 = BP_Flooding(y_hat,n,m,k,N_c,N_v,q,delta,no_iter)
	
	# SSR1 = an indicator variable representing whether BP algorithm recovered x correctly or not (when k is unknown)
	SSR1 = int(np.array_equal(I_hat1,I))
	# SSR2 = an indicator variable representing whether BP algorithm recovered x correctly or not (when k is known)
	SSR2 = int(np.array_equal(I_hat2,I))
	# FNR1 = false negative rate (when k is unknown)
	FNR1 = len(np.setdiff1d(I,I_hat1))/k
	# FPR1 = false positive rate (when k is unknown)
	FPR1 = len(np.setdiff1d(I_hat1,I))/(n-k)
	# FNR2 = false negative rate (when k is known)
	FNR2 = len(np.setdiff1d(I,I_hat2))/k
	# FPR2 = false positive rate (when k is known)
	FPR2 = len(np.setdiff1d(I_hat2,I))/(n-k)
	
	return [SSR1, SSR2, FNR1, FPR1, FNR2, FPR2]

start_time = time.time()
results = Parallel(n_jobs=-1)(delayed(CodeRunner)(trial) for trial in range(no_trials))
print("--- %s seconds ---" % (time.time() - start_time))

SSR1 = np.mean(np.array([results[i][0] for i in range(no_trials)]))
SSR2 = np.mean(np.array([results[i][1] for i in range(no_trials)]))
FNR1 = np.mean(np.array([results[i][2] for i in range(no_trials)]))
FPR1 = np.mean(np.array([results[i][3] for i in range(no_trials)]))
FNR2 = np.mean(np.array([results[i][4] for i in range(no_trials)]))
FPR2 = np.mean(np.array([results[i][5] for i in range(no_trials)]))

print('Success probability for unknown k = {}'.format(SSR1))
print('Success probability for known k = {}'.format(SSR2))
print('Unknown k: False negative rate = {}, False positive rate = {}'.format(FNR1,FPR1))
print('Known k: False negative rate = {}, False positive rate = {}'.format(FNR2,FPR2))