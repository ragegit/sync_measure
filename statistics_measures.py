# run simulation several times:
import numpy as np
import itertools as it
import random as rn
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import cm
from create_matrix_SR_SI_AR_AI import* # functions to creat matrices or change existing ones:
# SR(N,Tmax),SI(N,Tmax),AR(N,Tmax),AI(N,Tmax),
# SR_noisy(N,Tmax),AR_noisy(N,Tmax)
# mixed(N,Tmax,popspikes)
# add_noise(raster), smaller_step(raster)
from measures import *
from rasterplotMatrix import *

def gen_matrices(indices, N, Tmax):
	return {func: func(N,Tmax) for func in indices}

def generate_table(dictionary,xlist,ylist,functor,fmt,sep):
	#firstline, labels are to be printed here
	string = '\t'+'\t'.join([func.__name__ for func in xlist])+'\n'
	#now iterate for the rest of the lines
	#prints the indexes name and the corresponding return values of the functor applied to the contents of the dictionary,
	#tab-seperated and formatted by the formatsring fmt
	for yfunc in ylist:
		string += yfunc.__name__ + '\t' + '\t'.join(map(fmt.format,[functor(dictionary[xfunc][yfunc]) for xfunc in xlist])) + '\n'
	return string

def list_from_csv(filename):
	data = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		for row in reader:
			data.append(row)
	return data
	
raster_types = [SR,AR,SI,AI,AR_pattern,drift,osc,mixed5,mixed20,Poisson,smaller_step_SR,smaller_step_noisy_SR]
measurelist = [Renart_measure, my_measure, fano_factor]
measurement_dict = {fun: {
	func: [] for func in raster_types
	} for fun in measurelist}

N = 10
Tmax = 20000
num_iterations=10

for i in xrange(num_iterations):
	# different raster plots named by their features
	rasters = gen_matrices(raster_types, N,Tmax)
	for raster_type in raster_types:
		for measurement in measurelist:
			measurement_dict[measurement][raster_type].append(measurement(rasters[raster_type]))

f = open('output_var_N' + String(N) + '.csv', 'w')
f.write(generate_table(measurement_dict,measurelist,raster_types,np.var,r'{}',','))
f.close()

f = open('output_mean' + String(N) + '.csv', 'w')
f.write(generate_table(measurement_dict,measurelist,raster_types,np.mean,r'{}',','))
f.close()

