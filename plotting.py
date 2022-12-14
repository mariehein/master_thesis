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
small = True

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

def plot_start(small):
	if small:
		plt.figure(figsize=(5,3.75))
	else: 
		plt.figure()

def plot_end(name, ylim=1e5, small=True): 
	x = np.linspace(0.00001, 1, 10000)
	plt.plot(x, 1 / x, color="black", linestyle="--", label="random")

	plt.legend()
	plt.grid()
	plt.yscale("log")
	plt.ylim(1, ylim)
	plt.xlim(0,1)

	plt.ylabel(r"1/$\epsilon_B$")
	plt.xlabel(r"$\epsilon_S$")
	if small:
		plt.subplots_adjust(bottom=0.15, left= 0.19, top = 0.92, right = 0.965)

	#plt.title("IAD distortion")
	if envelope:
		plt.savefig("plots/combined/roc_.pdf")
	else:	
		plt.savefig("plots/combined/roc_"+name+".pdf")
	plt.show()

#Standard setup
small = False
plot_start(small)
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/4_default/supervised/")
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/4_default/cwola/")
name = "reproduction"
plot_end(name, small=small)


small = True
plot_start(small)
import_and_plot("roc_cwola_averaging", label="model",color="C0",folder="results/4_default/cwola/", N_runs=1)
name = "example"
plot_end(name, small=small)

#2% Signal setup
"""
import_and_plot("roc_supervised_averaging", label="Supervised",color="green",folder="results/4_default/supervised/")
import_and_plot("roc_IAD_averaging", label="IAD",color="grey",folder="results/4_signal20/IAD/")
import_and_plot("roc_cwola_averaging", label="CWoLa",color="orange",folder="results/4_signal20/cwola/")
import_and_plot("roc_cathode_averaging", label="CATHODE", color="deeppink", folder="results/4_signal20/cathode/")
"""

#IAD signal change
plot_start(small)
import_and_plot("roc_IAD_averaging", label="2%",color="C0",folder="results/4_signal20/IAD/")
import_and_plot("roc_IAD_averaging", label="0.64% (default)",color="C1",folder="results/4_default/IAD/")
import_and_plot("roc_IAD_averaging", label="0.4%",color="C2",folder="results/4_signal04/IAD/")
import_and_plot("roc_IAD_averaging", label="0.2%",color="C3",folder="results/4_signal02/IAD/")
import_and_plot("roc_IAD_averaging", label="0.1%",color="C4",folder="results/4_signal01/IAD/")
name = "signal_IAD"
plot_end(name, small=small)

#cwola signal change
plot_start(small)
import_and_plot("roc_cwola_averaging", label="2%",color="C0",folder="results/4_signal20/cwola/")
import_and_plot("roc_cwola_averaging", label="0.64% (default)",color="C1",folder="results/4_default/cwola/")
import_and_plot("roc_cwola_averaging", label="0.4%",color="C2",folder="results/4_signal04/cwola/")
import_and_plot("roc_cwola_averaging", label="0.2%",color="C3",folder="results/4_signal02/cwola/")
import_and_plot("roc_cwola_averaging", label="0.1%",color="C4",folder="results/4_signal01/cwola/")
name = "signal_cwola"
plot_end(name, small=small)

#cathode signal change
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="2%",color="C0",folder="results/4_signal20/cathode/")
import_and_plot("roc_cathode_averaging", label="0.64% (default)",color="C1",folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="0.4%",color="C2",folder="results/4_signal04/cathode/")
import_and_plot("roc_cathode_averaging", label="0.2%",color="C3",folder="results/4_signal02/cathode/")
import_and_plot("roc_cathode_averaging", label="0.1%",color="C4",folder="results/4_signal01/cathode/")
name = "signal_cathode"
plot_end(name, small=small)

#Change coupling layers
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="$N_c$=40", color="C0", folder="results/cathode/cathode_RealNVP_avg/", N_runs=1)
import_and_plot("roc_cathode_averaging", label="$N_c$=10", color="C1", folder="results/cathode_worse_DE/10_coupling/")
import_and_plot("roc_cathode_averaging", label="$N_c$=5", color="C2", folder="results/cathode_worse_DE/5_coupling/")
import_and_plot("roc_cathode_averaging", label="$N_c$=2", color="C3", folder="results/cathode_worse_DE/2_coupling/")
name = "cathode_improvement"
plot_end(name, small=small)

#Change coupling layers DE performance
small = True
plot_start(small)
import_and_plot("roc_DE_averaging", label="$N_c$=40, AUC=0.509", color="C0", folder="results/cathode/densities_RealNVP/", N_runs=1, start_runs=4)
import_and_plot("roc_DE_averaging", label="$N_c$=10, AUC=0.517", color="C1", folder="results/cathode_worse_DE/10_coupling/", N_runs=1)
import_and_plot("roc_DE_averaging", label="$N_c$=5, AUC=0.535", color="C2", folder="results/cathode_worse_DE/5_coupling/", N_runs=1)
import_and_plot("roc_DE_averaging", label="$N_c$=2, AUC=0.625", color="C3", folder="results/cathode_worse_DE/2_coupling/", N_runs=1)
name = "DE_improvement"
plot_end(name, small=small, ylim=1e2)

#val weight tests
small = False
plot_start(small)
import_and_plot("roc_cathode_averaging", label="Default", color="C0", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="No val weights", color="C1", folder="results/cathode/cathode_no_val_weights/")
import_and_plot("roc_cathode_averaging", label="No val oversampling", color="C2", folder="results/cathode/cathode_no_val_os/")
name="4_val_weights"
plot_end(name, small=small)

# RealNVP sampling methods
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="weight averaging", color="C0", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="ensemble sampling", color="C1", folder="results/cathode/cathode_RealNVP_ens/")
import_and_plot("roc_cathode_averaging", label="last epoch", color="C2", folder="results/cathode/cathode_RealNVP_no/")
name = "averaging_RealNVP"
plot_end(name, small=small)

# MAF RealNVP compare
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="RealNVP", color="C0", folder="results/cathode/cathode_RealNVP_every_run/")
import_and_plot("roc_cathode_averaging", label="MAF", color="C1", folder="results/cathode/cathode_MAF_every_run/")
name = "DE_influence"
plot_end(name, small=small)

# RealNVP retrain
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="Single DE", color="C0", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode_averaging", label="Retrain DE", color="C1", folder="results/cathode/cathode_RealNVP_every_run/")
name = "RealNVP_retrain"
plot_end(name, small=small)

# Averaging, no averaging
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="Averaging", color="C0", folder="results/cathode/cathode_RealNVP_avg/")
import_and_plot("roc_cathode", label="No averaging", color="C1", folder="results/cathode/cathode_RealNVP_avg/")
name = "effect_averaging"
plot_end(name, small=small)

# MAF stuff
small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="Single DE", color="C0", folder="results/cathode/cathode_MAF_avg/")
import_and_plot("roc_cathode_averaging", label="Retrain DE", color="C1", folder="results/cathode/cathode_MAF_every_run/")
name = "MAF_retrain"
plot_end(name, small=small)

small = True
plot_start(small)
import_and_plot("roc_cathode_averaging", label="weight averaging", color="C0", folder="results/cathode/cathode_MAF_avg/")
import_and_plot("roc_cathode_averaging", label="ensemble sampling", color="C1", folder="results/cathode/cathode_MAF_ens/")
import_and_plot("roc_cathode_averaging", label="last epoch", color="C2", folder="results/cathode/cathode_MAF_no/")
name= "averaging_MAF"
plot_end(name, small=small)

#IAD distortion
small = False
plot_start(small)
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0$",color="grey",folder="results/4_default/IAD/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.05$",color="C0",folder="results/4_distortion/gaussian_005/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.08$",color="C1",folder="results/4_distortion/gaussian_008/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.10$",color="C2",folder="results/4_distortion/gaussian_01/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.12$",color="C2",folder="results/4_distortion/gaussian_012/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.15$",color="C3",folder="results/4_distortion/gaussian_015/")
import_and_plot("roc_IAD_averaging", label=r"$\sigma=0.20$",color="C4",folder="results/4_distortion/gaussian_02/")
name = "IAD_distortion"
plot_end(name, small=small)




