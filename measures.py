import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
import brian as br

# Helper functions

def poprate(raster,upper,lower): # calculate the population rate
	return float(np.sum(np.sum(raster.T[upper:lower])))/(np.shape(raster.T[upper:lower])[0]*np.shape(raster.T[upper:lower])[1])

# actual measurement functions
# signature is always fun(raster) -> np.float
# one can calculate different measures for synchrony given the raster with neuron number = row number and time step = column number

def Renart_measure(raster):
	mat = np.corrcoef(x=raster, bias=1)
	return 1./(len(mat)*len(mat))*sum(sum(mat))
	
#def Renart_measure_alternative(raster):
	#cp_raster = []
	#Tmax = np.shape(raster)[1]
	#N = np.shape(raster)[0]
	#for n in xrange(N):
		#if(raster[n,:]!=np.zeros(Tmax) and raster[n,:]!=np.ones(Tmax)):
			#cp_raster.append(raster[n,:])
	#cp_raster = np.array(cp_raster)
	#mat = np.corrcoef(x=cp_raster, bias=1)
	#return 1./(len(mat)*len(mat))*sum(sum(mat))

def spike_var(raster):
	Tmax = np.shape(raster)[1]
	N = np.shape(raster)[0]
	bins = np.sum(raster, axis=0)
	return np.var(bins)
	
def fano_factor(raster): # coefficient of variation of the numper of spikes in each time bin = Fano factor
	return np.var(np.sum(raster,axis=0))/np.mean(np.sum(raster,axis=0))	

def poisson_measure(raster): # poisson scaled CV_si = Fano factor
	Tmax = np.shape(raster)[1]
	bins = np.sum(raster, axis=0)
	total = np.sum(raster)
	return np.var(bins)/(float(total)/(Tmax)) # ???
	
def my_measure(raster): # normalized Fano factor
	N = np.shape(raster)[0]
	bins = np.sum(raster, axis=0)
	spikemean = np.mean(bins)
	normalize = np.sqrt(spikemean/N*(N-spikemean)**2+(1-spikemean/N)*spikemean**2)
	m = np.std(bins)/normalize
	return m
	
def cv_isi(raster):
	isi = np.array([[]])
	for n in xrange(np.size(raster[:,0])):
		delta = np.diff(np.where(raster[n]==1))
		isi = np.concatenate((isi,delta), axis=1)
	return np.std(isi)/np.mean(isi)
	
#def cv_isi(raster):
	#cv_isi = []
	#for n in xrange(np.size(raster[:,0])):
		#delta = np.diff(np.where(raster[n]==1))
		#if 
		#isi = np.concatenate((isi,delta), axis=1)
	#return np.mean(cv_isi)