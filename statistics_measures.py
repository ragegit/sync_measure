# run simulation several times:
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
from measures import *
from rasterplotMatrix import *

list_names = ['SR','AR','SI','AI','M5','M20','P','smaller_step_SR','smaller_step_noisy_SR','smaller_step_more_noisy_SR']
measurelist = [Renart_measure, my_measure, poisson_measure]
measurement_dict = {fun.__name__ : {
	listname + 'list' : [] for listname in list_names
	} for fun in measurelist}

# Renart
rSRlist = []
smaller_step_rSRlist = []
smaller_step_noisy_rSRlist = []
smaller_step_more_noisy_rSRlist = []
rARlist = []
rSIlist = []
rAIlist = []
rM5list = []
rM20list = []
rPlist = []

# my_measure
mSRlist = []
smaller_step_mSRlist = []
smaller_step_noisy_mSRlist = []
smaller_step_more_noisy_mSRlist = []
mARlist = []
mSIlist = []
mAIlist = []
mM5list = []
mM20list = []
mPlist = []

# poisson_measure
pSRlist = []
smaller_step_pSRlist = []
smaller_step_noisy_pSRlist = []
smaller_step_more_noisy_pSRlist = []
pARlist = []
pSIlist = []
pAIlist = []
pM5list = []
pM20list = []
pPlist = []

N = 100
Tmax = 200
t=0

while(t<40):

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
	
	# Renart
	rSRlist.append(rSR)
	smaller_step_rSRlist.append(smaller_step_rSR)
	smaller_step_noisy_rSRlist.append(smaller_step_noisy_rSR)
	smaller_step_more_noisy_rSRlist.append(smaller_step_more_noisy_rSR)
	rARlist.append(rAR)
	rSIlist.append(rSI)
	rAIlist.append(rAI)
	rM5list.append(rM5)
	rM20list.append(rM20)
	rPlist.append(rP)
	
	# my_measure
	mSRlist.append(mSR)
	smaller_step_mSRlist.append(smaller_step_mSR)
	smaller_step_noisy_mSRlist.append(smaller_step_noisy_mSR)
	smaller_step_more_noisy_mSRlist.append(smaller_step_more_noisy_mSR)
	mARlist.append(mAR)
	mSIlist.append(mSI)
	mAIlist.append(mAI)
	mM5list.append(mM5)
	mM20list.append(mM20)
	mPlist.append(mP)

	# poisson_measure
	pSRlist.append(pSR)
	smaller_step_pSRlist.append(smaller_step_pSR)
	smaller_step_noisy_pSRlist.append(smaller_step_noisy_pSR)
	smaller_step_more_noisy_pSRlist.append(smaller_step_more_noisy_pSR)
	pARlist.append(pAR)
	pSIlist.append(pSI)
	pAIlist.append(pAI)
	pM5list.append(pM5)
	pM20list.append(pM20)
	pPlist.append(pP)
	t = t+1
	
print( np.var(rSRlist) )
print( np.var(smaller_step_rSRlist) )
print( np.var(smaller_step_noisy_rSRlist) )
print( np.var(smaller_step_more_noisy_rSRlist) )
print( np.var(rARlist) )
print( np.var(rSIlist) )
print( np.var(rAIlist) )
print( np.var(rM5list) )
print( np.var(rM20list) )
print( np.var(rPlist) )
	
print( np.var(mSRlist) )
print( np.var(smaller_step_mSRlist) )
print( np.var(smaller_step_noisy_mSRlist) )
print( np.var(smaller_step_more_noisy_mSRlist) )
print( np.var(mARlist) )
print( np.var(mSIlist) )
print( np.var(mAIlist) )
print( np.var(mM5list) )
print( np.var(mM20list) )
print( np.var(mPlist) )
	
print( np.var(pSRlist) )
print( np.var(smaller_step_pSRlist) )
print( np.var(smaller_step_noisy_pSRlist) )
print( np.var(smaller_step_more_noisy_pSRlist) )
print( np.var(pARlist) )
print( np.var(pSIlist) )
print( np.var(pAIlist) )
print( np.var(pM5list) )
print( np.var(pM20list) )
print( np.var(pPlist) )

print( np.mean(rSRlist) )
print( np.mean(smaller_step_rSRlist) )
print( np.mean(smaller_step_noisy_rSRlist) )
print( np.mean(smaller_step_more_noisy_rSRlist) )
print( np.mean(rARlist) )
print( np.mean(rSIlist) )
print( np.mean(rAIlist) )
print( np.mean(rM5list) )
print( np.mean(rM20list) )
print( np.mean(rPlist) )
	
print( np.mean(mSRlist) )
print( np.mean(smaller_step_mSRlist) )
print( np.mean(smaller_step_noisy_mSRlist) )
print( np.mean(smaller_step_more_noisy_mSRlist) )
print( np.mean(mARlist) )
print( np.mean(mSIlist) )
print( np.mean(mAIlist) )
print( np.mean(mM5list) )
print( np.mean(mM20list) )
print( np.mean(mPlist) )
	
print( np.mean(pSRlist) )
print( np.mean(smaller_step_pSRlist) )
print( np.mean(smaller_step_noisy_pSRlist) )
print( np.mean(smaller_step_more_noisy_pSRlist) )
print( np.mean(pARlist) )
print( np.mean(pSIlist) )
print( np.mean(pAIlist) )
print( np.mean(pM5list) )
print( np.mean(pM20list) )
print( np.mean(pPlist) )
