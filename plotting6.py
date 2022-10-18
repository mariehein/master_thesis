#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:45:30 2021

@author: mariehein
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#import densities_fns as fn

envelope=False
plot_all=False

def interpolate(x,x_array,y_array):
	ind = np.where(x_array > x)[0]#[0]
	if ind.size==0 or ind[0]==0:
		return y_array[0]
	ind=ind[0]
	return (y_array[ind]-y_array[ind-1])/(x_array[ind]-x_array[ind-1])*(x-x_array[ind-1])+y_array[ind-1]

def mean_and_std(x_arrays,y_arrays,number=50000):
	x_values = x_arrays[0]#np.linspace(0,1,number)
	y_values = np.zeros((len(y_arrays),number))
	for j in range(len(y_arrays)):
		y_values[j]=interp1d(x_arrays[j],y_arrays[j])(x_values)
	y_mean = np.median(y_values,axis=0)
	y_std = np.std(y_values,axis=0) 
	return x_values, y_mean, y_std

def min_and_max(x_arrays,y_arrays,number=50000):
	x_values = np.logspace(-7,0,number)
	y_values = np.zeros((len(y_arrays),len(x_values)))
	print(x_arrays.shape)
	print(y_arrays.shape)
	print(y_values.shape)
	for j in range(len(y_arrays)):
		y_values[j]=interp1d(x_arrays[j],y_arrays[j])(x_values)
	y_mean = np.median(1/y_values,axis=0)
	y_max = 1/np.min(y_values,axis=0) 
	y_min = 1/np.max(y_values,axis=0)
	return x_values, y_mean, y_min, y_max

def mean_and_percentiles(x_arrays,y_arrays,number=50000):
	x_values = np.logspace(-7,0,number)
	y_values = np.zeros((len(y_arrays),len(x_values)))
	for j in range(len(y_arrays)):	
		y_values[j] = interp1d(x_arrays[j], y_arrays[j])(x_values)
	
	"""
	for j in range(len(y_arrays)):
		for i,x in enumerate(x_values):
			y_values[j][i]=interp1d(x,x_arrays[j],y_arrays[j])
	"""
	py = 1/y_values	
	y_mean = np.median(py,axis=0)
	y_max = np.percentile(py,84,axis=0) 
	y_min = np.percentile(py,16,axis=0)
	return x_values, y_mean, y_min, y_max

def printing(x_arrays,y_arrays,number=50000):
	x_values = np.array([0.2,0.4,0.6,0.8])
	y_values = np.zeros((len(y_arrays),len(x_values)))
	for j in range(len(y_arrays)):	
		y_values[j] = interp1d(x_arrays[j], y_arrays[j])(x_values)
	
	"""
	for j in range(len(y_arrays)):
		for i,x in enumerate(x_values):
			y_values[j][i]=interp1d(x,x_arrays[j],y_arrays[j])
	"""
	py = 1/y_values	
	y_mean = np.median(py,axis=0)
	y_max = np.percentile(py,84,axis=0) 
	y_min = np.percentile(py,16,axis=0)
	return x_values, y_mean, y_min, y_max

def plot_fn(tpr,fpr,color,label):
	if envelope: 
		tpr_mean, fpr_mean, fpr_min, fpr_max= min_and_max(tpr,fpr)
		plt.plot(tpr_mean, fpr_mean, color=color,label=label)
		plt.fill_between(tpr_mean, fpr_min, fpr_max, alpha=0.2, facecolor=color)
	else:
		tpr_mean, fpr_mean, fpr_min, fpr_max= mean_and_percentiles(tpr,fpr)
		#print(printing(tpr, fpr))
		plt.plot(tpr_mean, fpr_mean, color=color,label=label)
		plt.fill_between(tpr_mean, fpr_min, fpr_max, alpha=0.2, facecolor=color)
		"""tpr_mean, fpr_mean, fpr_std = mean_and_std(tpr,fpr) 
		per_fpr_std = fpr_std/fpr_mean**2
		plt.fill_between(tpr_mean, 1/fpr_mean-per_fpr_std, 1/fpr_mean+per_fpr_std, alpha=0.2, facecolor=color)
		plt.plot(tpr_mean, 1/fpr_mean, color=color,label=label)"""
	return 

def plt_collected(tpr,fpr,x,y,color,label):
	plt.figure()
	plt.title("ROCs for "+label)
	for i in range(0,len(fpr)):
		plt.plot(tpr[i],1/fpr[i],color=color)
	plt.plot(x,y,'.',color="black")
	plt.grid()
	plt.yscale("log")
	plt.ylim(1, 1e5)
	plt.xlim(0,1)
	x = np.linspace(0.00001, 1, 10000)
	plt.plot(x, 1 / x, color="black", linestyle="--", label="random")

	plt.ylabel(r"1/$\epsilon_B$")
	plt.xlabel(r"$\epsilon_S$")
	plt.savefig("plots/combined/rocs_"+label+".pdf")
	return

def import_and_plot(path, label=None, color="grey", folder="rocs/", N_runs =10, start_runs =0):	
	if label is None: 
		label=path
	fpr=np.load(folder+"fpr_"+path+".npy")[start_runs:start_runs+N_runs]
	tpr=np.load(folder+"tpr_"+path+".npy")[start_runs:start_runs+N_runs]
	plot_fn(tpr,fpr,color,label)
	return tpr,fpr

def import_and_plot_min(path, label=None, color="grey", folder="rocs/"):
	if label is None: 
		label=path
	fpr=1-np.load(folder+"fpr_"+path+".npy")
	tpr=np.load(folder+"tpr_"+path+".npy")
	plot_fn(tpr,fpr,color,label)
	return tpr,fpr


x = [0.2,0.4,0.6,0.8]
y_supervised = [6.67e3, 9.19e2, 2.21e2, 4.53e1]
y_cwola = [1.68e3, 2.41e2, 3.79e1, 3.24]
y_cathode = [4.41e3, 5.01e2, 4.69e1, 3.24]

plt.figure()

#Standard setup
"""
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/4_default/cwola/")
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/4_default/supervised/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/cathode/cathode_RealNVP_avg/")
"""

#6 Inputs Standard setup
"""
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/6_default/supervised/")
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/6_default/IAD/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/6_default/cathode/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/6_default/cwola/")
"""

#6 Hyperparameters IAD

import_and_plot("roc_IAD_averaging", label="4 inputs",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_IAD_averaging", label="Default hyperparameters",color="C0",folder="results/6_default/IAD/")
import_and_plot("roc_IAD_averaging", label="IAD hyperparameters",color="C1",folder="results/hyperparameter_runs/IAD_full/")


#6 Inputs Hyperparameters
"""
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/hyperparameter_runs/supervised/")
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/hyperparameter_runs/IAD_full/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/hyperparameter_runs/cathode/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/hyperparameter_runs/cwola/")
"""

#6 Hyperparameters CATHODE
"""
import_and_plot("roc_cathode_averaging", label="4 inputs",color="grey",folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="Default hyperparameters",color="C0",folder="results/6_default/cathode/")
import_and_plot("roc_cathode_averaging", label="IAD hyperparameters",color="C1",folder="results/hyperparameter_runs/cathode_full/")
import_and_plot("roc_cathode_averaging", label="CATHODE hyperparameters",color="C2",folder="results/hyperparameter_runs/cathode/")
"""

#6 Hyperparameters CWoLa
"""
import_and_plot("roc_cwola_averaging", label="4 inputs",color="grey",folder="results/4_default/cwola/")
import_and_plot("roc_cwola_averaging", label="Default hyperparameters",color="C0",folder="results/6_default/cwola/")
import_and_plot("roc_cwola_averaging", label="IAD hyperparameters",color="C1",folder="results/hyperparameter_runs/cwola_full/")
import_and_plot("roc_cwola_averaging", label="CWoLa hyperparameters",color="C2",folder="results/hyperparameter_runs/cwola/")
"""

#6 Hyperparameters Supervised
"""
import_and_plot("roc_supervised_averaging", label="4 inputs",color="grey",folder="results/4_default/supervised/")
import_and_plot("roc_supervised_averaging", label="Default hyperparameters",color="C0",folder="results/6_default/supervised/")
import_and_plot("roc_supervised_averaging", label="IAD hyperparameters",color="C1",folder="results/hyperparameter_runs/supervised_full/")
import_and_plot("roc_supervised_averaging", label="Supervised hyperparameters",color="C2",folder="results/hyperparameter_runs/supervised/")
"""


#Gaussian inputs
"""
import_and_plot("roc_IAD_averaging", label="default",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_IAD_averaging", label="1G",color="C0",folder="results/gaussian_inputs/gauss1/")
import_and_plot("roc_IAD_averaging", label="2G",color="C1",folder="results/gaussian_inputs/gauss2/")
#import_and_plot("roc_IAD_averaging", label="3G",color="C1",folder="results/gaussian_inputs/gauss3/")
import_and_plot("roc_IAD_averaging", label="5G",color="C2",folder="results/gaussian_inputs/gauss5/")
import_and_plot("roc_IAD_averaging", label="7G",color="C3",folder="results/gaussian_inputs/gauss7/")
import_and_plot("roc_IAD_averaging", label="10G",color="C4",folder="results/gaussian_inputs/gauss10/")
"""

#Gaussian inputs supervised
"""
import_and_plot("roc_supervised_averaging", label="default",color="green",folder="results/4_default/supervised/")
#import_and_plot("roc_supervised_averaging", label="1G",color="C0",folder="results/supervised_gaussian/gauss1/")
#import_and_plot("roc_supervised_averaging", label="2G",color="C1",folder="results/supervised_gaussian/gauss2/")
#import_and_plot("roc_supervised_averaging", label="5G",color="C2",folder="results/supervised_gaussian/gauss5/")
#import_and_plot("roc_supervised_averaging", label="7G",color="C3",folder="results/supervised_gaussian/gauss7/")
import_and_plot("roc_supervised_averaging", label="10G",color="C4",folder="results/supervised_gaussian/gauss10/")
"""

#Gauss, IAD compare
"""
import_and_plot("roc_IAD_averaging", label="default",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_IAD_averaging", label=r"$\tau_{32}$",color="C0",folder="results/6_default/IAD/")
import_and_plot("roc_IAD_averaging", label="2G",color="C1",folder="results/gaussian_inputs/gauss2/")
"""







#plt.plot(x,y_cathode, '.',color="black")#deeppink mediumvioletred
#plt.plot(x,y_cwola,'.',color="xkcd:pumpkin")
#plt.plot(x,y_supervised,'.',color="limegreen")

x = np.linspace(0.00001, 1, 10000)
plt.plot(x, 1 / x, color="black", linestyle="--", label="random")

plt.legend()
plt.grid()
plt.yscale("log")
plt.ylim(1, 1e5)
plt.xlim(0,1)

plt.ylabel(r"1/$\epsilon_B$")
plt.xlabel(r"$\epsilon_S$")

#plt.title("IAD distortion")
if envelope:
	plt.savefig("plots/combined/roc_.pdf")
else:	
	plt.savefig("plots/combined/6inputs/roc_6_IAD.pdf")
plt.show()


