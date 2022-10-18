import numpy as np
import matplotlib.pyplot as plt

def plot_one_loss(loss, name, color):
	x = range(1,len(loss)+1)
	loss_avg = np.zeros(len(loss)-5)
	plt.plot(x, loss, linestyle=':', color=color)
	x = range(2,len(loss)-3)
	for i in range(2,len(loss)-3):
		loss_avg[i-2] = np.mean(loss[i-2:i+3])
	plt.plot(x, loss_avg, label=name, color=color)


def loss_plotting(loss1, loss2, name1, name2, save_to):
	plt.figure()

	l1,  =plt.plot(loss1, color="black", linestyle=":", label="Per epoch value", visible="False")
	l2, = plt.plot(loss1, color="black", label="5-epoch average", visible="False")

	plot_one_loss(loss1, name1, 'C0')	
	plot_one_loss(loss2, name2, 'C1')	
	
	plt.ylabel(r"loss")
	plt.xlabel(r"epochs")

	plt.legend()
	plt.grid()
	l1.remove()
	l2.remove()

	plt.savefig("plots/loss/"+save_to)
	plt.show()

folder = "results/6_default/cwola/run0/"
history = np.load(folder+"classifier_history.npy", allow_pickle=True).item()
loss_plotting(history['loss'],history['val_loss'], "Training", "Validation", "hyperparameters/loss_cwola_default.pdf" )
"""
bkg_loss = np.load(folder+"bkg_loss.npy")
sig_loss = np.load(folder+"sig_loss.npy")

loss_plotting(bkg_loss,sig_loss, "Background", "Signal", "loss_bkg_sig.pdf" )
"""
