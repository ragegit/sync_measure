import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
from matplotlib import cm
from create_matrix_SR_SI_AR_AI import* # functions to creat matrices or change existing ones:
# SR(N,Tmax),SI(N,Tmax),AR(N,Tmax),AI(N,Tmax),
# SR_noisy(N,Tmax),AR_noisy(N,Tmax)
# mixed(N,Tmax,popspikes)
# add_noise(raster), smaller_step(raster)
from measures import * # has functions to calculate measures defined by Renart, Brunel, ... in it:
# Renart_measure(raster), fano_factor, my_measure

#N=1000 # number of neurons
#Tmax=2000 # number of timesteps

def main(N,Tmax):
	# different raster plots named by their features--------------------------------------------------------------
	rasterSR = SR(N,Tmax)
	noisy_rasterSR = add_noise(rasterSR)
	smaller_step_rasterSR = smaller_step(rasterSR)
	smaller_step_noisy_rasterSR = add_noise(smaller_step_rasterSR)
	smaller_step_more_noisy_rasterSR = add_noise(smaller_step_noisy_rasterSR)
	rasterAR = AR(N,Tmax)

	rasterSI = SI(N,Tmax)
	rasterAI = AI(N,Tmax)

	Mixed5 = mixed(N,Tmax,5) # create a raster, which is mostly SI, but has 5 population spikes 
	Mixed10 = mixed(N,Tmax,10)
	Mixed20 = mixed(N,Tmax,20)
	rasterPo = Poisson(N,Tmax)
	
	#decreasing synchrony 	
	rasterlist = increase_noise(rasterSI, 20)
	crucial_rasters = [rasterlist[0],rasterlist[1],rasterlist[2],rasterlist[20]]
	Renart_measure_list = map(Renart_measure, crucial_rasters)
	fano_factor_list = map(fano_factor, crucial_rasters)
	my_measure_list = map(my_measure, crucial_rasters)
	
	plt.rcParams.update({'font.size':15})
	
	#plot measure outcomes along axis
	fig = plt.figure()

	a = fig.add_subplot(3,1,1)
	plt.plot(Renart_measure_list,len(Renart_measure_list)*[0], 'o')
	plt.title("Renart measure")

	a = fig.add_subplot(3,1,2)
	plt.plot(fano_factor_list,len(fano_factor_list)*[0], 'o')
	plt.title("fano factor")

	a = fig.add_subplot(3,1,3)
	plt.plot(my_measure_list,len(my_measure_list)*[0], 'o')
	plt.title("My measure")

	#plot rasters--------------------------------------------------------------
	plt.matshow(rasterSI[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Synchronous Irregular state")
	plt.savefig('Pictures/SI')
	
	plt.matshow(rasterlist[1][:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Some synchrony left from the Synchronous Irregular state")
	plt.savefig('Pictures/noisy_SI_1')
	
	plt.matshow(rasterlist[2][:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Some synchrony left from the Synchronous Irregular state")
	plt.savefig('Pictures/noisy_SI_2')
	
	plt.matshow(rasterlist[20][:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("No synchrony left from the Synchronous Irregular state")
	plt.savefig('Pictures/noisy_SI_3')
	
	plt.matshow(rasterSR[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Synchronous Regular state")
	plt.savefig('Pictures/SR')
	plt.matshow(smaller_step_rasterSR[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Synchronous Regular state with smaller time bin")
	plt.savefig('Pictures/small_bin_SR')
	plt.matshow(smaller_step_noisy_rasterSR[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Smaller time bin - still SR?")
	plt.savefig('Pictures/noisy_SR')
	plt.matshow(smaller_step_more_noisy_rasterSR[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("very noisy SR")
	plt.savefig('Pictures/very_noisy_SR')
	plt.matshow(rasterAI[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Asynchronous Irregular state")
	plt.savefig('Pictures/AI')
	plt.matshow(rasterAR[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Asynchronous Regular state")
	plt.savefig('Pictures/AR')
	plt.matshow(Mixed5[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Mixed state")
	plt.savefig('Pictures/Mixed5')
	plt.matshow(Mixed10[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Mixed state")
	plt.savefig('Pictures/Mixed10')
	plt.matshow(Mixed20[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.title("Mixed state")
	plt.ylabel("neuron number")
	plt.savefig('Pictures/Mixed20')
	plt.matshow(rasterPo[:,0:200],cmap=cm.binary)
	plt.xlabel("time bin")
	plt.title("Poisson state")
	plt.ylabel("neuron number")
	plt.savefig('Pictures/Poisson')
	plt.show()

	## plot rasters in subplot--------------------------------------------------------------
	fig = plt.figure(2000)

	a = fig.add_subplot(2,2,1)
	plt.plot(np.where(rasterSR[:,0:200]==1)[0],np.where(rasterSR[:,0:200]==1)[1], '|')
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Synchronous Regular state")

	a = fig.add_subplot(2,2,2)
	plt.plot(np.where(rasterAR[:,0:200]==1)[0],np.where(rasterAR[:,0:200]==1)[1], '|')
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Asynchronous Regular state")

	a = fig.add_subplot(2,2,3)
	plt.plot(np.where(rasterSI[:,0:200]==1)[0],np.where(rasterSI[:,0:200]==1)[1], '|')
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Synchronous Irregular state")

	a = fig.add_subplot(2,2,4)
	plt.plot(np.where(rasterAI[:,0:200]==1)[0],np.where(rasterAI[:,0:200]==1)[1], '|')
	plt.xlabel("time bin")
	plt.ylabel("neuron number")
	plt.title("Asynchronous Irregular state")

	plt.show()

	## evaluate data with different measures------------------------------------------------------------------------
	## population rate
	## timebin as large as the step width
	#popSR0 = poprate(rasterSR,0,1) # calculation of pop. rate for a timebin as large as the step width for SR
	#popSR1 = poprate(rasterSR,1,2)
	#popSR2 = poprate(rasterSR,2,3)
	#popSR6 = poprate(rasterSR,6,7)

	#popAR0 = poprate(rasterAR,0,1) # calculation of pop. rate for a timebin as large as the step width for AR
	#popAR1 = poprate(rasterAR,1,2)
	#popAR2 = poprate(rasterAR,2,3)
	#popAR6 = poprate(rasterAR,6,7)

	#popSR0 = poprate(rasterSR,0,1) # calculation of pop. rate for a timebin as large as the step width for SR
	#popSR1 = poprate(rasterSR,1,2)
	#popSR2 = poprate(rasterSR,2,3)
	#popSR6 = poprate(rasterSR,6,7)

	#popAI0 = poprate(rasterAI,0,1) # calculation of pop. rate for a timebin as large as the step width for AI
	#popAI1 = poprate(rasterAI,1,2)
	#popAI2 = poprate(rasterAI,2,3)
	#popAI6 = poprate(rasterAI,6,7)

	## timebin as large as the step width
	#ratelistSR1 = map(poprate,[rasterSR]*Tmax,range(0,Tmax-1),range(1,Tmax)) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR1 = map(poprate,[rasterAR]*Tmax,range(0,Tmax-1),range(1,Tmax)) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI1 = map(poprate,[rasterSI]*Tmax,range(0,Tmax-1),range(1,Tmax))
	#ratelistAI1 = map(poprate,[rasterAI]*Tmax,range(0,Tmax-1),range(1,Tmax))

	## timebin twice as large as the step width
	#x = range(0,Tmax-1,2)
	#y = range(2,Tmax,2)
	#B = len(x)
	#ratelistSR2 = map(poprate,[rasterSR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR2 = map(poprate,[rasterAR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI2 = map(poprate,[rasterSI]*B,x,y)
	#ratelistAI2 = map(poprate,[rasterAI]*B,x,y)

	## timebin six times as large as the step width
	#x = range(0,Tmax-5,6)
	#y = range(6,Tmax,6)
	#B = len(x)
	#ratelistSR6 = map(poprate,[rasterSR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR6 = map(poprate,[rasterAR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI6 = map(poprate,[rasterSI]*B,x,y)
	#ratelistAI6 = map(poprate,[rasterAI]*B,x,y)

	## plot population rates
	#map(plt.plot, [ratelistSR1,ratelistAR1,ratelistSI1,ratelistAI1], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()
	#map(plt.plot, [ratelistSR2,ratelistAR2,ratelistSI2,ratelistAI2], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()
	#map(plt.plot, [ratelistSR6,ratelistAR6,ratelistSI6,ratelistAI6], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()
	## Mixed and AI state should be compared!


	## timebin as large as the step width
	#ratelistSR1 = map(poprate,[rasterSR]*Tmax,range(0,Tmax-1),range(1,Tmax)) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR1 = map(poprate,[rasterAR]*Tmax,range(0,Tmax-1),range(1,Tmax)) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI1 = map(poprate,[rasterSI]*Tmax,range(0,Tmax-1),range(1,Tmax))
	#ratelistAI1 = map(poprate,[rasterAI]*Tmax,range(0,Tmax-1),range(1,Tmax))

	## timebin twice as large as the step width
	#x = range(0,Tmax-1,2)
	#y = range(2,Tmax,2)
	#B = len(x)
	#ratelistSR2 = map(poprate,[rasterSR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR2 = map(poprate,[rasterAR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI2 = map(poprate,[rasterSI]*B,x,y)
	#ratelistAI2 = map(poprate,[rasterAI]*B,x,y)

	## timebin six times as large as the step width
	#x = range(0,Tmax-5,6)
	#y = range(6,Tmax,6)
	#B = len(x)
	#ratelistSR6 = map(poprate,[rasterSR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for SR
	#ratelistAR6 = map(poprate,[rasterAR]*B,x,y) # calculation of pop. rate for a timebin as large as the step width for AR
	#ratelistSI6 = map(poprate,[rasterSI]*B,x,y)
	#ratelistAI6 = map(poprate,[rasterAI]*B,x,y)

	# plot population rates
	#map(plt.plot, [ratelistSR1,ratelistAR1,ratelistSI1,ratelistAI1], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()
	#map(plt.plot, [ratelistSR2,ratelistAR2,ratelistSI2,ratelistAI2], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()
	#map(plt.plot, [ratelistSR6,ratelistAR6,ratelistSI6,ratelistAI6], ['o']*4)
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.show()

	#plt.rcParams.update({'font.size':10})

	##--------------------------------------------with subplots
	#fig = plt.figure(1000)

	#a = fig.add_subplot(2,2,1)
	#plt.plot(ratelistSR1, 'o')
	#plt.title("SR")
	#plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,2)
	#plt.plot(ratelistAR1, 'o')
	#plt.title("AR")
	##plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,3)
	#plt.plot(ratelistSI1, 'o')
	#plt.title("SI")
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,4)
	#plt.plot(ratelistAI1, 'o')
	#plt.title("AI")
	##plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#plt.savefig('Simulations1/step1')
	##plt.show()

	##--------------------------------------------with subplots
	#fig = plt.figure(1001)

	#a = fig.add_subplot(2,2,1)
	#plt.plot(ratelistSR2, 'o')
	#plt.title("SR")
	#plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,2)
	#plt.plot(ratelistAR2, 'o')
	#plt.title("AR")
	##plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,3)
	#plt.plot(ratelistSI2, 'o')
	#plt.title("SI")
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,4)
	#plt.plot(ratelistAI2, 'o')
	#plt.title("AI")
	##plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#plt.savefig('Simulations1/step2')
	##plt.show()

	##--------------------------------------------with subplots
	#fig = plt.figure(1002)

	#a = fig.add_subplot(2,2,1)
	#plt.plot(ratelistSR6, 'o')
	#plt.title("SR")
	#plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,2)
	#plt.plot(ratelistAR6, 'o')
	#plt.title("AR")
	##plt.ylabel("instantaneous population rate")
	##plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,3)
	#plt.plot(ratelistSI6, 'o')
	#plt.title("SI")
	#plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))

	#a.autoscale_view()
	#a = fig.add_subplot(2,2,4)
	#plt.plot(ratelistAI6, 'o')
	#plt.title("AI")
	##plt.ylabel("instantaneous population rate")
	#plt.xlabel("timestep")
	#plt.ylim((-0.1,1.1))
	#plt.savefig('Simulations1/step6')
	##plt.show()

	# Renart_measure
	rSR = Renart_measure(rasterSR)
	smaller_step_rSR = Renart_measure(smaller_step_rasterSR)
	smaller_step_noisy_rSR = Renart_measure(smaller_step_noisy_rasterSR)
	smaller_step_more_noisy_rSR = Renart_measure(smaller_step_more_noisy_rasterSR)
	rAR = Renart_measure(rasterAR)
	rSI = Renart_measure(rasterSI)
	rAI = Renart_measure(rasterAI)
	rM5 = Renart_measure(Mixed5)
	rM10 = Renart_measure(Mixed10)
	rM20 = Renart_measure(Mixed20)
	rP = Renart_measure(rasterPo)

	# my_measure
	mSR = my_measure(rasterSR)
	smaller_step_mSR = my_measure(smaller_step_rasterSR)
	smaller_step_noisy_mSR = my_measure(smaller_step_noisy_rasterSR)
	smaller_step_more_noisy_mSR = my_measure(smaller_step_more_noisy_rasterSR)
	mAR = my_measure(rasterAR)
	mSI = my_measure(rasterSI)
	mAI = my_measure(rasterAI)
	mM5 = my_measure(Mixed5)
	mM10 = my_measure(Mixed10)
	mM20 = my_measure(Mixed20)
	mP = my_measure(rasterPo)

	# poisson_measure
	pSR = poisson_measure(rasterSR)
	smaller_step_pSR = poisson_measure(smaller_step_rasterSR)
	smaller_step_noisy_pSR = poisson_measure(smaller_step_noisy_rasterSR)
	smaller_step_more_noisy_pSR = poisson_measure(smaller_step_more_noisy_rasterSR)
	pAR = poisson_measure(rasterAR)
	pSI = poisson_measure(rasterSI)
	pAI = poisson_measure(rasterAI)
	pM5 = poisson_measure(Mixed5)
	pM10 = poisson_measure(Mixed10)
	pM20 = poisson_measure(Mixed20)
	pP = poisson_measure(rasterPo)

	## print and compare Renart_measure and my_measure
	print("\t \t \t Renart \t my_measure \t poisson_measure")
	print("AR \t \t \t %f \t %f \t %f" %(rAR, mAR, pAR))
	print("SR \t \t \t %f \t %f \t %f" %(rSR, mSR, pSR))
	print("noisy, smaller step SR %f \t %f \t %f" %(smaller_step_noisy_rSR,smaller_step_noisy_mSR,smaller_step_noisy_pSR))
	print("more noisy, smaller step SR %f \t %f \t %f" %(smaller_step_more_noisy_rSR,smaller_step_more_noisy_mSR,smaller_step_more_noisy_pSR))
	print("AI \t \t \t %f \t %f \t %f" %(rAI,mAI,pAI))
	print("SI \t \t \t %f \t %f \t %f" %(rSI,mSI,pSI))
	print("Mixed 5 \t \t %f \t %f \t %f" %(rM5,mM5,pM5))
	print("Mixed 10 \t \t %f \t %f \t %f" %(rM10,mM10,pM10))
	print("Mixed 20 \t \t %f \t %f \t %f" %(rM20,mM20,pM20))
	print("Poisson \t \t %f \t %f \t %f" %(rP,mP,pP))
	return 0
