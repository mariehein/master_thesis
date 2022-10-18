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


minmass = [2.9+i*0.1 for i in range(9)]
maxmass = [3.3+i*0.1 for i in range(9)]
ssb_width = 0.2
BH_percentiles = [90., 99., 99.9, 99.99]
BH_reps = ["0.1", "0.01", "0.001", "0.0001"]

results = np.zeros((4,9))
true_results =  np.zeros((4,9))
rel_results =  np.zeros((4,9))
rel_error =  np.zeros((4,9))
name = "cathode_without"
cwola = True

cwola_rel = [0.09, 0.20, 0.21, 0.08]
cathode_rel = [0.10, 0.22, 0.40, 0.76]

folders = ["results/cathode_sliding_without/window"+str(i)+"/" for i in range(1,10)]

for i, f in enumerate(folders): 
	print("----------------------")
	print(f)
	if not cwola:
		samples_preds = np.load(f+"run0/samples_preds.npy")[400000:,1]
	full_preds = np.load(f+"run0/full_preds.npy")
	m_preds = np.load(f+"m_preds.npy")
	label_preds = np.load(f+"label_preds.npy")
	mask = (m_preds > minmass[i]) & (m_preds<maxmass[i])
	inner_preds = full_preds[mask] 
	if cwola:
		ssb_mask =  ~mask & (m_preds > minmass[i]-ssb_width) & (m_preds<maxmass[i]+ssb_width)
		samples_preds = full_preds[ssb_mask]
	inner_labels = label_preds[mask]
	bkg_preds = inner_preds[inner_labels==0]
	sig_preds = inner_preds[inner_labels==1]
	
	for j, perc in enumerate(BH_percentiles):
		print()
		eps = np.percentile(samples_preds, perc)
		if perc==0:
			eps=0.
		print("Threshold:", eps)
		N_samples_after = sum(samples_preds>eps)
		N_samples =len(samples_preds)
		print("Percentage samples:", N_samples_after/N_samples)
		N_after = sum(inner_preds>eps)
		N =len(inner_preds)
		print("Percentage data:", N_after/N)
		N_bkg = sum(bkg_preds>eps)
		print("Bkg:", N_bkg/len(bkg_preds))
		N_sig = sum(sig_preds>eps)
		print("Sig before:", len(sig_preds), "sig after:", N_sig)
		if cwola:		
			results[j,i] = (N_after-N*(100-perc)/100)/np.sqrt((100-perc)/100*N*2)
			rel_results[j,i] = (N_after-N*(100-perc)/100)/np.sqrt((100-perc)/100*N*2+ ((100-perc)/100*N*cwola_rel[j])**2)
		else:
			results[j,i] = (N_after-N*(100-perc)/100)/np.sqrt((100-perc)/100*N)
			rel_results[j,i] = (N_after-N*(100-perc)/100)/np.sqrt((100-perc)/100*N+ ((100-perc)/100*N*cathode_rel[j])**2)
		print("Significance:", results[j,i])
		print("Sys Significance:", rel_results[j,i])
		if cwola:
			true_results[j,i] = N_sig/np.sqrt(2*N_bkg)
		else:
			true_results[j,i] = N_sig/np.sqrt(N_bkg)
		print("True significance:", true_results[j,i])
		rel_error[j,i] = (N_after-N*(100-perc)/100)/((100-perc)/100*N)


print(np.mean(rel_error, axis=1))



plt.figure()
x = range(1,10)
plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
plt.axhline(0, color="black", label="0$\sigma$")
for j, perc in enumerate(BH_percentiles):
	plt.plot(x, results[j], label=r"1/$\epsilon_B$="+BH_reps[j])

plt.grid()


plt.ylabel(r"Significance")
plt.xlabel(r"Sliding window #")
plt.legend()

plt.savefig("plots/bh/bh_"+name+".pdf")

plt.figure()
x = range(1,10)
plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
plt.axhline(0, color="black", label="0$\sigma$")
for j, perc in enumerate(BH_percentiles):
	plt.plot(x, rel_results[j], label=r"1/$\epsilon_B$="+BH_reps[j])

plt.grid()


plt.ylabel(r"Significance")
plt.xlabel(r"Sliding window #")
plt.legend()

plt.savefig("plots/bh/bh_"+name+"_rel.pdf")


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

	plt.savefig("plots/bh/bh_"+name+"_"+str(perc)+".pdf")


