#functions to create binary matrices where the row gives the discretized time and the column is the neuron id 
import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
import brian as br
import copy as cp

# Helper functions

def smaller_step(raster):
	newraster = np.insert(raster,np.arange(0,len(raster[0])),0,axis=1)
	return newraster

#def shuffle(raster, n): # Choses n spikes in a rasterplot and moves them to a random spot. Then the rasterplot is appended in a list. The procedure continues until all spikes where moved
	#rasterlist = [raster]
	#idx_spike = list(np.array(np.where(raster==1)).T)
	#idx_silence = list(np.array(np.where(raster==0)).T)
	#N = min(len(idx_spike),len(idx_silence))
	#iters = int(N/n)-1
	#for j in xrange(iters):
		#for i in xrange(n):
			#if(len(idx_spike)!=0 and len(idx_silence)!=0):
				#spike = rn.randint(0,len(idx_spike)-1)
				#silence = rn.randint(0,len(idx_silence)-1)
				#raster[idx_spike[spike][0]][idx_spike[spike][1]] = 0
				#raster[idx_silence[silence][0]][idx_silence[silence][1]] = 1
				#idx_spike.pop(spike)
				#idx_silence.pop(silence)
			#else: break
		#rasterlist.append(raster)
	#return rasterlist
	
def shuffle(raster, n): # Choses n spikes in a rasterplot and moves them to a random spot. Then the rasterplot is appended in a list. The procedure continues until all spikes where moved
	rasterlist = [raster]
	cp_raster = cp.copy(raster)
	idx_spike = list(np.array(np.where(raster==1)).T)
	idx_silence = list(np.array(np.where(raster==0)).T)
	N = min(len(idx_spike),len(idx_silence))
	for j in xrange(N):
		if(len(idx_spike)!=0 and len(idx_silence)!=0):
			spike = rn.randint(0,len(idx_spike)-1)
			silence = rn.randint(0,len(idx_silence)-1)
			cp_raster[idx_spike[spike][0]][idx_spike[spike][1]] = 0
			cp_raster[idx_silence[silence][0]][idx_silence[silence][1]] = 1
			idx_spike.pop(spike)
			idx_silence.pop(silence)
		if((j%n)==0):
			print j
			rasterlist.append(cp_raster)
	return rasterlist
	
	
def add_noise(raster):
	newraster = np.zeros(np.shape(raster))
	idx = np.where(raster==1)
	newidx = [[],[]]
	i=0
	for element in idx[1]:
		x = np.random.randint(3)
		if(x==2):
			element=(element+1)%len(raster[0])
		elif(x==1):
			element=(element-1)%len(raster[0])
		newidx[1].append(element)
		newidx[0].append(idx[0][i])
		i=i+1
	i=0
	for element in newidx[0]:
		newraster[element][newidx[1][i]] = 1
		i=i+1
	return newraster

def mixed(N,Tmax,popspikes):
	mat = np.floor(1./9*np.random.randint(10, size=(N,Tmax)))
	number = int((float(Tmax))/popspikes)
	for j in xrange(N):
		for i in xrange(Tmax):
			if(i%number == 0):
				x = np.random.randint(4)
				if(x==2 or x==1 or x==0):
					mat[j][i] = 1
				else:
					mat[j][i] = 0
	return mat
	
######resorting from synchronous to asynchronous starting from a synchronous raster:

def increase_noise(raster, iteration): # to go from synchronous to poissonian
	rasterlist = [raster]
	for i in xrange(iteration):
		raster = add_noise(raster)
		rasterlist.append(raster)
	return rasterlist
# use map to see measure for all rasters
	
	
# actual raster funktions
# signature is func(N,Tmax) -> raster
def SR(N,Tmax):
	arr = np.zeros(Tmax, dtype='int')
	liste = []
	for i in xrange(Tmax):
		if(i%6==0):
			liste.append(i)
	arr[liste] = 1
	arr = np.array(N*[arr])
	return arr
	
def SR_noisy(N,Tmax):
	mat = []
	for j in xrange(N):
		arr = np.zeros(Tmax, dtype='int')
		liste = []
		for i in xrange(Tmax):
			if(i%6==0):
				liste.append(i)
		x = np.random.randint(3)
		if(x==2):
			liste[np.random.randint(len(liste))]+=1
		elif(x==1):
			liste[np.random.randint(len(liste))]-=1		
		arr[liste] = 1
		mat.append(arr)
	mat = np.array(mat)
	return mat

def AR_pattern(N,Tmax):
	mat = []
	k = 0
	for j in xrange(N):
		arr = np.zeros(Tmax, dtype='int')
		liste = []
		for i in xrange(Tmax):
			if(i%6==0):
				liste.append((i+k)%(Tmax-2))
		k+=2
		arr[liste] = 1
		mat.append(arr)
	mat = np.array(mat)
	return mat
	
def AR_noisy(N,Tmax):
	mat = []
	k = 0
	for j in xrange(N):
		arr = np.zeros(Tmax, dtype='int')
		liste = []
		for i in xrange(Tmax):
			if(i%6==0):
				liste.append((i+k)%(Tmax-2))
		k+=2
		x = np.random.randint(3)
		if(x==2):
			liste[np.random.randint(len(liste))]+=1
		elif(x==1):
			liste[np.random.randint(len(liste))]-=1		
		arr[liste] = 1
		mat.append(arr)
	mat = np.array(mat)
	return mat
	
def AR(N,Tmax):
	arr = np.random.randint(1, 100, size=N)
	mat = np.zeros((N,Tmax), dtype='int')
	for n in xrange(N):
		num = arr[n]
		for t in xrange(1,Tmax):
			if(t%num == 0):
				mat[n][t] = 1
	return mat
	
def SI(N,Tmax):
	arr = np.zeros(Tmax)
	t = 0
	while(t<Tmax):
		t = t + rn.expovariate(0.03)
		if(t<Tmax):
			arr[np.int(t)] = 1
	return np.array(N*[arr])
	
def SI_uniform(N,Tmax):
	arr = np.random.randint(2, size=Tmax)
	return np.tile(arr,(N,1))

def AI(N,Tmax):
	return np.floor(1./99*np.random.randint(10, size=(N,Tmax)))
	#return np.random.randint(2, size=(N,Tmax))
def pair_corr(N,Tmax):
	mat = np.zeros((N,Tmax), dtype='int')
	mat[:][1]=1  # allow calculation of correlation coefficient
	n1 = np.random.randint(N)
	n2 = np.random.randint(N)
	for t in xrange(Tmax):
		if(t%5==0):
			mat[n1][t] = 1
			mat[n2][t] = 1
	return mat
	
def rand_pair_corr(N,Tmax):
	mat = np.zeros((N,Tmax), dtype='int')
	mat[:][1]=1 # allow calculation of correlation coefficient
	for t in xrange(Tmax):
		if(t%5==0):
			n1 = np.random.randint(N)
			n2 = np.random.randint(N)
			mat[n1][t] = 1
			mat[n2][t] = 1
	return mat
	
def higher_order_corr(N,Tmax):
	mat = np.zeros((N,Tmax), dtype='int')
	mat[:][1]=1 # allow calculation of correlation coefficient
	for t in xrange(Tmax):
		if(t%10==0):
			n1 = np.random.randint(N)
			n2 = np.random.randint(N)
			n3 = np.random.randint(N)
			n4 = np.random.randint(N)
			mat[n1][t] = 1
			mat[n2][t] = 1
			mat[n3][t] = 1
			mat[n4][t] = 1
	return mat
	
	
def Poisson(N,Tmax):
	return np.floor(1./999*np.random.randint(1000, size=(N,Tmax)))

def drift(N,Tmax):
	mat = np.zeros((N,Tmax), dtype='int')
	for n in xrange(N):
		for t in xrange(Tmax):
			x = np.random.randint(Tmax)
			if(x<t):
				mat[n][t] = 1
	return mat

def osc(N,Tmax):
	mat = np.zeros((N,Tmax), dtype='int')
	for n in xrange(N):
		for t in xrange(Tmax):
			x = np.random.randint(Tmax)
			if(x< Tmax*np.sin(t)):
				mat[n][t] = 1
	return mat
	
	
def mixed5(N,Tmax):
	return mixed(N,Tmax,5)
def mixed10(N,Tmax):
	return mixed(N,Tmax,10)
def mixed20(N,Tmax):
	return mixed(N,Tmax,20)

def smaller_step_SR(N, Tmax):
	return smaller_step(SR(N,Tmax))
	
def smaller_step_noisy_SR(N, Tmax):
	return add_noise(smaller_step(SR(N,Tmax)))
	
def smaller_step_more_noisy_SR(N, Tmax):
	return add_noise(add_noise(smaller_step(SR(N,Tmax))))
	


