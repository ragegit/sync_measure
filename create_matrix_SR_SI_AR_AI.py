#functions to create binary matrices where the row gives the discretized time and the column is the neuron id 
import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
import brian as br

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

def AR(N,Tmax):
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
	
def SI(N,Tmax):
	arr = np.random.randint(2, size=Tmax)
	return np.tile(arr,(N,1))

def AI(N,Tmax):
	return np.floor(1./9*np.random.randint(10, size=(N,Tmax)))
	#return np.random.randint(2, size=(N,Tmax))
	
def Poisson(N,Tmax):
	return np.floor(1./999*np.random.randint(1000, size=(N,Tmax)))
	
def smaller_step(raster):
	newraster = np.insert(raster,np.arange(0,len(raster[0])),0,axis=1)
	return newraster
	
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
