import numpy as np
import classifier_utils as cl
import MAF_utils as MAF
import RealNVP_utils as RNVP
import dataprep_utils as dp
import bumphunt_utils as bh
import plotting_utils as pf
import argparse
import os
import matplotlib.pyplot as plt



BH_percentiles = [90., 99., 99.9, 99.99]#,99.,99.9, 99.99]
BH_reps = ["0.1", "0.01", "0.001", "0.0001"]

results = np.zeros((4,9))
true_results =  np.zeros((4,9))
rel_results =  np.zeros((4,9))
rel_error =  np.zeros((4,9))
folder = "results/k_fold/cathode_sliding_without/"
name= "cathode_without"
cwola = False

cwola_rel = [0.05, 0.12, 0.10, 0.23]
cathode_rel = [0.06, 0.11, 0.20, 0.15]

for k in range(9):
	f = folder+"window"+str(k+1)+"/"
	print(f)

	N_samples_after = np.zeros((4,5))
	N_samples = np.zeros((4,5))
	N_after = np.zeros((4,5))
	N = np.zeros((4,5))
	N_bkg = np.zeros((4,5))
	N_sig = np.zeros((4,5))
	N_bkg_orig = 0

	for i in range(5): 

		samples_preds = np.load(f+"run"+str(i)+"/samples_preds.npy")[:,1]
		full_preds = np.load(f+"run"+str(i)+"/preds_averaged.npy")
		Y_test = np.load(f+"run"+str(i)+"/Y_test.npy")
		length = sum(Y_test[:,1]==0)
		inner_preds = np.concatenate((full_preds[Y_test[:,1]==0], full_preds[Y_test[:,1]==1]))
		inner_labels = np.concatenate((np.zeros(length),np.ones(len(inner_preds)-length)))
		bkg_preds = inner_preds[inner_labels==0]
		sig_preds = inner_preds[inner_labels==1]

		N_bkg_orig += len(bkg_preds)
		for j, perc in enumerate(BH_percentiles):
			eps = np.percentile(samples_preds, perc)
			if perc==0:
				eps=0.
			N_samples_after[j,i] = sum(samples_preds>eps)
			N_samples[j,i] =len(samples_preds)
			N_after[j,i] = sum(inner_preds>eps)
			N[j,i] =len(inner_preds)
			N_bkg[j,i] = sum(bkg_preds>eps)
			N_sig[j,i] = sum(sig_preds>eps)
	N_samples_after = np.sum(N_samples_after, axis=1)
	N_samples = np.sum(N_samples, axis=1)
	N_after = np.sum(N_after, axis=1)
	N = np.sum(N, axis=1)
	N_bkg = np.sum(N_bkg, axis=1)
	N_sig = np.sum(N_sig, axis=1)
	print(N_after)
	print(N_sig)
	print(N_bkg)
	print()
	for j, perc in enumerate(BH_percentiles):
		print()
		print(perc)
		print("Samples:", N_samples_after[j]/N_samples[j],";Bkg:", N_bkg[j]/N_bkg_orig,  "; Data:", N_after[j]/N[j])
		if cwola:
			results[j,k] = (N_after[j]-N[j]*(100-perc)/100)/np.sqrt((100-perc)/100*N[j]*6)
			rel_results[j,k] = (N_after[j]-N[j]*(100-perc)/100)/np.sqrt((100-perc)/100*N[j]*6+((100-perc)/100*cwola_rel[j]*N[j])**2)
		else:
			results[j,k] = (N_after[j]-N[j]*(100-perc)/100)/np.sqrt((100-perc)/100*N[j])
			rel_results[j,k] = (N_after[j]-N[j]*(100-perc)/100)/np.sqrt((100-perc)/100*N[j]+((100-perc)/100*cathode_rel[j]*N[j])**2)
		print("Significance:", results[j,k])
		print("Sys Significance:", rel_results[j,k])
		if cwola:
			true_results[j,k] = N_sig[j]/np.sqrt(6*N_bkg[j])
		else:
			true_results[j,k] = N_sig[j]/np.sqrt(N_bkg[j])
		print("True significance:", true_results[j,k])
		rel_error[j,k] = (N_after[j]-N[j]*(100-perc)/100)/((100-perc)/100*N[j])


print(np.mean(rel_error, axis=1))


plt.figure()
x = range(1,10)
plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
plt.axhline(0, color="black", label="0$\sigma$")
for j, perc in enumerate(BH_percentiles):
	plt.plot(x, results[j], label=r"$\epsilon_B$="+BH_reps[j])

plt.grid()


plt.ylabel(r"Significance")
plt.xlabel(r"Sliding window #")
plt.legend()

plt.savefig("plots/bh_k_fold/bh_"+name+".pdf")


plt.figure()
x = range(1,10)
plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
plt.axhline(0, color="black", label="0$\sigma$")
for j, perc in enumerate(BH_percentiles):
	plt.plot(x, rel_results[j], label=r"$\epsilon_B$="+BH_reps[j])

plt.grid()


plt.ylabel(r"Significance")
plt.xlabel(r"Sliding window #")
plt.legend()

plt.savefig("plots/bh_k_fold/bh_"+name+"_rel.pdf")


for j, perc in enumerate(BH_percentiles):
	plt.figure()
	x = range(1,10)

	plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
	plt.axhline(0, color="black", label="0$\sigma$")

	plt.plot(x, true_results[j], label="True Significance")
	plt.plot(x, results[j], label="Inferred Significance")

	plt.grid()

	plt.legend()

	plt.ylabel(r"Significance")
	plt.xlabel(r"Sliding window #")

	plt.savefig("plots/bh_k_fold/bh_"+name+"_"+str(perc)+".pdf")




