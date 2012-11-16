import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
import brian as br

# one can calculate different measures for synchrony given the raster with neuron number = row number and time step = column number

def poprate(raster,upper,lower): # calculate the population rate
	return float(np.sum(np.sum(raster.T[upper:lower])))/(np.shape(raster.T[upper:lower])[0]*np.shape(raster.T[upper:lower])[1])

def Renart_measure(raster):
	mat = np.corrcoef(x=raster, bias=1)
	return 1./(len(mat)*len(mat))*sum(sum(mat))
	
def spike_var(raster):
	Tmax = np.shape(raster)[1]
	N = np.shape(raster)[0]
	bins = np.sum(raster, axis=0)
	return np.var(bins)
	
def CV_si(raster): # coefficient of variation of the numper of spikes in each time bin
	return np.var(np.sum(raster,axis=0))/np.mean(np.sum(raster,axis=0))	

def poisson_measure(raster): # poisson scaled CV_si
	Tmax = np.shape(raster)[1]
	bins = np.sum(raster, axis=0)
	total = np.sum(raster)
	return np.var(bins)/(float(total)/(Tmax)) # ???
	
def my_measure(raster): # normalized CV_si
	N = np.shape(raster)[0]
	bins = np.sum(raster, axis=0)
	spikemean = np.mean(bins)
	normalize = np.sqrt(spikemean/N*(N-spikemean)**2+(1-spikemean/N)*spikemean**2)
	m = np.std(bins)/normalize
	return m
	
def ISI(raster):
	liste = []
	for j in xrange(len(raster)):
		idx = np.where(raster[j]==1)[0]
		for i in xrange(len(idx)-1):
			liste.append(idx[i+1]-idx[i])
	return liste
	
def CV_ISI(raster):
	liste = ISI(raster)
	histo = [liste.count(x) for x in xrange(np.max(liste)+1)] # list for a SII histogram
	return np.std(histo)/np.mean(histo)
	