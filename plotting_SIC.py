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
background=False

sys_err = 0.2*np.sqrt(120000)
sys_err = 0

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
		if not background:
			plt.plot(tpr_mean, tpr_mean*np.sqrt(fpr_mean+sys_err**2*fpr_mean**2), color=color,label=label)
			plt.fill_between(tpr_mean, tpr_mean*np.sqrt(fpr_min+sys_err**2*fpr_min**2), tpr_mean*np.sqrt(fpr_max+sys_err**2*fpr_max**2), alpha=0.2, facecolor=color)
		else:
			plt.plot(1/fpr_mean, tpr_mean/np.sqrt(1/fpr_mean+sys_err**2/fpr_mean**2), color=color,label=label)
			plt.fill_between(1/fpr_mean, tpr_mean/np.sqrt(1/fpr_min+sys_err**2/fpr_min**2), tpr_mean/np.sqrt(1/fpr_max+sys_err**2/fpr_max**2), alpha=0.2, facecolor=color)		
		"""tpr_mean, fpr_mean, fpr_std = mean_and_std(tpr,fpr) 
		per_fpr_std = fpr_std/fpr_mean**2
		plt.fill_between(tpr_mean, 1/fpr_mean-per_fpr_std, 1/fpr_mean+per_fpr_std, alpha=0.2, facecolor=color)
		plt.plot(tpr_mean, 1/fpr_mean, color=color,label=label)"""
	return 

def plt_collected(tpr,fpr,x,y,color,label):
	plt.figure()
	plt.title("ROCs for "+label)
	for i in range(0,len(fpr)):
		#plt.plot(tpr[i],tpr[i]/np.sqrt(fpr[i]),color=color)
		plt.plot(1/fpr[i],tpr[i]/np.sqrt(fpr[i]),color=color)
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
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/4_default/supervised/")
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/4_default/cwola/")
"""

import_and_plot("roc_cwola_averaging", label="model", color="C0", folder="results/4_default/cwola/", N_runs=1)

#CATHODE
"""
import_and_plot("roc_cathode_averaging", label="Default", color="C0", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="Run for sliding window", color="C1", folder="results/cathode_sliding/window5/")
import_and_plot("roc_test_avg", label="k-fold cross validation", color="C2", folder="results/k_fold/cathode_sliding/window5/")
"""

#CATHODE Signal
"""
import_and_plot("roc_test_avg", label="0.64% (default)", color="C0", folder="results/k_fold/cathode_sliding/window5/")
import_and_plot("roc_test_avg", label="0.4%", color="C1", folder="results/k_fold/cathode_sliding_04/window5/")
import_and_plot("roc_test_avg", label="0.3%", color="C2", folder="results/k_fold/cathode_sliding_03/window5/")
"""

#CWoLa
"""
import_and_plot("roc_cwola_averaging", label="Default", color="C0", folder="results/4_default/cwola/")
import_and_plot("roc_cwola_averaging", label="Run for sliding window", color="C1", folder="results/cwola_sliding/window5/")
import_and_plot("roc_test_avg", label="k-fold cross validation", color="C2", folder="results/k_fold/cwola_sliding/window5/")
"""

#CWoLa Signal
"""
import_and_plot("roc_test_avg", label="1.0%", color="C3", folder="results/k_fold/cwola_sliding_10/window5/")
import_and_plot("roc_test_avg", label="0.64% (default)", color="C0", folder="results/k_fold/cwola_sliding/window5/")
import_and_plot("roc_test_avg", label="0.3%", color="C2", folder="results/k_fold/cwola_sliding_03/window5/")
"""

#plt.plot(x,y_cathode, '.',color="black")#deeppink mediumvioletred
#plt.plot(x,y_cwola,'.',color="xkcd:pumpkin")
#plt.plot(x,y_supervised,'.',color="limegreen")

x = np.linspace(0.00001, 1, 10000)
plt.plot(x, np.sqrt(x), color="black", linestyle="--", label="random")

plt.legend()
plt.grid()
if background:
	plt.xscale("log")
	plt.xlim(1e-5,1)
else:
	plt.xlim(0,1)
plt.ylim(0, 10)


plt.ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
if background:
	plt.xlabel(r"$\epsilon_B$")
else:
	plt.xlabel(r"$\epsilon_S$")

#plt.title("IAD distortion")
plt.savefig("plots/combined/SIC_example.pdf")#bumphunt/sic_cwola_signal_fpr.pdf")
plt.show()


