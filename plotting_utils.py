import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
from scipy.stats import gaussian_kde
import warnings


output_dim = 256
reg = 0.01

def conditional_plot_two(
    array1, array2, title="plots/default.pdf", name1="array1", name2="array2", N_bins=50, logit=True
):
    limits = np.array([[-4.5, 0], [-8, 0.5], [-3, 3], [-3, 3]])
    names = ["m_1", "m_2", "tau_1", "tau_2"]

    f, axes = plt.subplots(2, 4)
    f.set_size_inches(10, 7.5)
    f.suptitle("lower left and blue: "+name1+", top right and orange: "+name2)

    for i in range(4):
        axes[1, i].hist2d(
            array2[:, i+1], array2[:, 0], bins=(N_bins, N_bins), cmap=plt.cm.viridis
        )
        axes[0, i].hist2d(
            array1[:, i+1], array1[:, 0], bins=(N_bins, N_bins), cmap=plt.cm.viridis
        )

        #axes[i, j].set_title(name1)
        #axes[j, i].set_title(name2)

        axes[0, i].set_xlabel(names[i])
        axes[0, i].set_ylabel("m_inv")
        axes[1, i].set_xlabel(names[i])
        axes[1, i].set_ylabel("m_inv")

        if logit:
            #axes[0, i].set_xlim(limits[i, 0], limits[i, 1])
            axes[0, i].set_xlim(limits[i, 0], limits[i, 1])
            #axes[1, i].set_xlim(limits[i, 0], limits[i, 1])
            axes[1, i].set_xlim(limits[i, 0], limits[i, 1])
    f.tight_layout()
    f.savefig(title)
    plt.show()
    return



def features_plot_two(
    array1, array2, title="plots/default.pdf", name1="array1", name2="array2", N_bins=50, logit=True
):
    limits = np.array([[-4.5, 0], [-8, 0.5], [-3, 3], [-3, 3]])
    names = ["m_1", "m_2", "tau_1", "tau_2"]

    f, axes = plt.subplots(2, 4)
    f.set_size_inches(10, 7.5)
    f.suptitle("lower left and blue: "+name1+", top right and orange: "+name2)

    for i in range(4):
        axes[1, i].hist(
            (array1[:,i],array2[:, i]), bins=N_bins,density=True
        )
        axes[0, i].hist(
            array1[:, i], bins=N_bins,density=True
        )

        #axes[i, j].set_title(name1)
        #axes[j, i].set_title(name2)

        axes[0, i].set_xlabel(names[i])
        #axes[0, i].set_ylabel("m_inv")
        axes[1, i].set_xlabel(names[i])
        #axes[1, i].set_ylabel("m_inv")

        if logit:
            #axes[0, i].set_xlim(limits[i, 0], limits[i, 1])
            axes[0, i].set_xlim(limits[i, 0], limits[i, 1])
            #axes[1, i].set_xlim(limits[i, 0], limits[i, 1])
            axes[1, i].set_xlim(limits[i, 0], limits[i, 1])
    f.tight_layout()
    f.savefig(title)
    plt.show()
    return

def densities_plot(array, title="plots/default.pdf", N_bins=50):
    size = len(array[0])
    f, axes = plt.subplots(size, size)
    f.set_size_inches(20, 15)

    for i in range(size):
        for j in range(i):
            axes[i, j].hist2d(
                array[:, i], array[:, j], bins=(N_bins, N_bins), cmap=plt.cm.viridis
            )
        axes[i, i].hist(array[:, i], bins=N_bins, density=True)
    f.savefig(title)
    plt.show()
    return


def densities_plot_two(
    array1, array2, title="plots/default.pdf", name1="array1", name2="array2", N_bins=50, logit=True, names=None
):
    size=len(array1[0])
    limits = np.array([[-3, 3]]*size)
    if names is None:
        names = ["m_1", "m_2", "tau21_1", "tau21_2", "tau_32_1", "tau32_2"]+["other"]*(size-6)
        names = names[:size]
    
    f, axes = plt.subplots(size, size)
    f.set_size_inches(10, 7.5)
    f.suptitle("lower left and blue: "+name1+", top right and orange: "+name2)

    for i in range(size):
        for j in range(i):
            axes[j, i].hist2d(
                array2[:, i], array2[:, j], bins=(N_bins, N_bins), cmap=plt.cm.viridis
            )
            axes[i, j].hist2d(
                array1[:, i], array1[:, j], bins=(N_bins, N_bins), cmap=plt.cm.viridis
            )

            #axes[i, j].set_title(name1)
            #axes[j, i].set_title(name2)

            axes[i, j].set_xlabel(names[i])
            axes[i, j].set_ylabel(names[j])
            axes[j, i].set_xlabel(names[i])
            axes[j, i].set_ylabel(names[j])

            if logit:
                axes[i, j].set_xlim(limits[i, 0], limits[i, 1])
                axes[i, j].set_ylim(limits[j, 0], limits[j, 1])
                axes[j, i].set_xlim(limits[i, 0], limits[i, 1])
                axes[j, i].set_ylim(limits[j, 0], limits[j, 1])

        axes[i, i].hist(
            (array1[:, i], array2[:, i]),
            bins=N_bins,
            density=True,
            label=(name1, name2),
        )
        if logit:
            axes[i, i].set_xlim(limits[i, 0], limits[i, 1])
        axes[i, i].set_xlabel(names[i])
    f.tight_layout()
    f.savefig(title)
    plt.show()
    return

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def plot_training(history,title="training", directory=None):
	plt.figure()
	plt.plot(history.history["loss"])
	#plt.plot(history.history["val_loss"])
	plt.title(title)
	#plt.legend(["train", "validation"], loc="upper right")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	if directory is None:
		plt.savefig("plots/training/"+title+"_train.pdf")
	else:
		plt.savefig(directory+"training_train.pdf")
	plt.show()
    
	plt.figure()
	#plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title(title)
	#plt.legend(["train", "validation"], loc="upper right")
	plt.ylabel("val loss")
	plt.xlabel("epoch")
	if directory is None:
		plt.savefig("plots/training/"+title+"_val.pdf")
	else:
		plt.savefig(directory+"training_val.pdf")
	plt.show()
	return

def make_one_array(twod_arr,new_arr):
	if len(new_arr) < len(twod_arr.T):
		app=np.zeros(len(twod_arr.T)-len(new_arr),dtype=None)
		new_arr=np.concatenate((new_arr,app),axis=0)
	elif len(twod_arr.T) < len(new_arr):
		app=np.zeros((len(twod_arr),len(new_arr)-len(twod_arr.T)),dtype=None)
		twod_arr=np.concatenate((twod_arr,app),axis=1)
	return np.concatenate((twod_arr,np.array([new_arr])),axis=0)

def plot_roc(test_results, test_labels, title="roc", directory = None, direc_run = None):
	fpr, tpr, thresholds = roc_curve(test_labels, test_results)

	auc = roc_auc_score(test_labels, test_results)

	x = np.linspace(0.001, 1, 10000)
	plt.figure()
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
		plt.plot(tpr, 1 / fpr, label="model")
	plt.plot(x, 1 / x, color="black", linestyle="--", label="random")

	plt.legend()
	plt.grid()
	plt.yscale("log")
	plt.ylim(1, 1e5)
	plt.xlim(0,1)

	plt.ylabel(r"1/$\epsilon_B$")
	plt.xlabel(r"$\epsilon_S$")

	plt.title(title)

	if directory is None:
		directory ="rocs/"
	if direc_run is None:
		direc_run=directory
	
	plt.savefig(direc_run+title+"roc.pdf")

	if Path(directory+"tpr_"+title+".npy").is_file():
		tpr_arr=np.load(directory+"tpr_"+title+".npy")
		np.save(directory+"tpr_"+title+".npy",make_one_array(tpr_arr,tpr))
		fpr_arr=np.load(directory+"fpr_"+title+".npy")
		np.save(directory+"fpr_"+title+".npy",make_one_array(fpr_arr,fpr))
	else: 
		np.save(directory+"tpr_"+title+".npy",np.array([tpr]))
		np.save(directory+"fpr_"+title+".npy",np.array([fpr]))
	
	if directory=="rocs/":	
		f=open("aucs/"+title+".txt",'a+')
	else:
		f=open(directory+title+".txt",'a+')

	f.write("\n"+str(auc))
	return auc

def plot_scores(results, labels, title="scores", name0="background", name1="signal", directory=None):
	plt.figure()
	plt.title(title)
	plt.hist([results[labels==1],results[labels==0]],50,label=[name1,name0],density=True)
	plt.xlabel("classifier score")
	plt.legend()
	if directory is None:
		plt.savefig("plots/scores/"+title+".pdf")
	else:
		plt.savefig(directory+"scores"+title+".pdf")
	return 
