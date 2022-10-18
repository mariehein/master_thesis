import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotting_utils as pf
import yaml
from keras.utils.np_utils import to_categorical 

models = keras.models
layers = keras.layers
regularizers = keras.regularizers

def prediction_averaging_sequential(files, X, partial_smoothing=False):
	model_list = []

	for i,file_i in enumerate(files):
		if partial_smoothing:
			model_list = np.append(model_list, models.load_model(file_i,custom_objects={'partially_noisy_loss': partially_noisy_loss}))
		else:
			model_list = np.append(model_list, models.load_model(file_i))

	results = np.array([model.predict(X) for model in model_list])
	#print(results.shape)

	return np.mean(results,axis=0)

def take_best(folder, history, best=10):
	inds = np.argsort(history.history["val_loss"])+1
	files = []	
	print("\nBest epochs: ", inds[:best])
	for i in inds[:best]:
		files = np.append(files, folder+"%02d.hdf5" % (i,))
	return files

def partially_noisy_loss(y_true, y_pred, sample_weight= None):
	cce_norm = keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
	cce_smooth = keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE,label_smoothing=0.3)
	return tf.reduce_sum(cce_norm(y_true,y_pred,sample_weight)*y_true[:,0]+cce_smooth(y_true,y_pred,sample_weight)*y_true[:,1])/float(len(y_true))

def make_model(activation="relu",hidden=3,inputs=4,lr=1e-3,dropout=0.1, l1=0, l2 =0, momentum = 0.9, label_smoothing=0):
	model = models.Sequential()
	model.add(layers.Dense(64,input_shape=(inputs,)))
	for i in range(hidden-1):
		if activation =="relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
	model.add(layers.Dense(2, activation="softmax"))

	loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	model.compile(
		loss=loss,
		optimizer=keras.optimizers.Adam(lr, beta_1=momentum),
		metrics=["accuracy"],
	)

	return model



def classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds=None, direc_run=None, DE=False, samples=None):
	if direc_run is None:	
		direc_run=args.directory
	
	with open(args.cl_filename, 'r') as stream:
		params = yaml.safe_load(stream)

	model = make_model(activation=params['activation'], hidden=int(params['hidden']), inputs=args.inputs, lr=float(params['lr']), dropout=float(params['dropout']), l1=float(params['l1']), l2 =float(params['l2']), momentum = float(params['beta_1']), label_smoothing=float(params['label_smoothing']))

	if not os.path.exists(direc_run):
		os.makedirs(direc_run)
	folder=direc_run+"models/"
	if not os.path.exists(folder):
		os.makedirs(folder)


	class LossHistory(keras.callbacks.Callback):

		def on_train_begin(self, logs={}):
		    self.loss_sig = np.zeros(params['epochs'])
		    self.loss_bkg = np.zeros(params['epochs'])
		    self.labels_signal = to_categorical(np.ones(int(sum(Y_test[:,1]))),2)
		    self.labels_bkg = to_categorical(np.zeros(int(sum(Y_test[:,0]))),2)

		def on_epoch_end(self, epoch, logs={}):
		    cce = keras.losses.CategoricalCrossentropy()
		    pred_test = self.model.predict(X_test)
		    self.loss_sig[epoch] = cce(self.labels_signal, pred_test[Y_test[:,1]==1]).numpy()
		    self.loss_bkg[epoch] = cce(self.labels_bkg, pred_test[Y_test[:,1]==0]).numpy()
		    print(epoch, self.loss_sig[epoch-1], self.loss_bkg[epoch-1], flush=True)

	checkpoint_filepath = folder+'{epoch:02d}.hdf5'
	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=False,
		monitor='val_loss',)
	callbacks=[model_checkpoint_callback]

	if args.cl_loss_tracking:
		hist = LossHistory()
		callbacks = [model_checkpoint_callback, hist]

	if args.use_class_weights:
		class_weight = {0: 1, 1: len(Y_train)/sum(Y_train.T[1])-1}
	else:
		class_weight = {0: 1, 1: 1}
		
	if args.use_val_weights:
		val_weight = {0: 1, 1: len(Y_val)/sum(Y_val.T[1])-1}
		val_sample_weights = val_weight[0]*Y_val[:,0]+val_weight[1]*Y_val[:,1]
	else:
		val_weight = {0: 1, 1: 1}
		val_sample_weights = np.ones(len(Y_val))

	print("\nTraining class weights: ", class_weight, "; Validation class weights: ", val_weight, "\n")
	
	results = model.fit(
		X_train,
		Y_train,
		batch_size=params['batchsize'],
		epochs=params['epochs'],
		shuffle=True,
		verbose=2,
		validation_data=(X_val, Y_val, val_sample_weights),
		class_weight=class_weight,
		callbacks=callbacks,
	)

	if args.cl_loss_tracking:
		plt.figure()
		#plt.plot(hist.loss_sig, label="signal")
		plt.plot(hist.loss_bkg, label="bkg")
		plt.title("Losses, not weighted")
		plt.legend()
		plt.ylabel("loss")
		plt.xlabel("epoch")
		plt.savefig(direc_run+"losses.pdf")
		plt.show()

		plt.figure()
		plt.plot(hist.loss_sig, label="signal")
		plt.title("Signal loss")
		plt.legend()
		plt.ylabel("loss")
		plt.xlabel("epoch")
		plt.savefig(direc_run+"signal_loss.pdf")
		plt.show()
		
		np.save(direc_run+"sig_loss.npy", hist.loss_sig)
		np.save(direc_run+"bkg_loss.npy", hist.loss_bkg)

	np.save(direc_run+'classifier_history.npy', results.history)
	
	files = take_best(folder, results, best=args.cl_N_best_epochs)
	np.save(direc_run+"best_files_classifier.npy", files)
	test_2 = prediction_averaging_sequential(files, X_test)[:,1]	
	
	if args.cl_detailed_plots:
		for f in files:
			model = models.load_model(f)
			pred = model.predict(X_test)[:,1]
			pf.plot_scores(pred, Y_test[:,1], title=f[len(folder):-5]+"_scores",directory = direc_run)
			pf.plot_roc(pred, Y_test[:,1],title=f[len(folder):-5]+"_roc", directory = direc_run)

	pf.plot_training(results, "training_classifier", directory=direc_run)

	test_results = model.predict(X_test).T[1]

	pf.plot_scores(test_results, Y_test[:,1], title="samples",directory=direc_run)
	pf.plot_scores(test_2, Y_test[:,1], title="samples_averaging",directory=direc_run)

	if DE:
		name = "DE"
	else:
		name = args.mode

	if args.signal_percentage !=0:
		print("AUC last epoch: %.3f" % pf.plot_roc(test_results, Y_test[:,1],title="roc_"+name,directory=args.directory, direc_run=direc_run))
		print("AUC with averaging: %.3f" % pf.plot_roc(test_2, Y_test[:,1], title="roc_"+name+"_averaging",directory=args.directory, direc_run=direc_run))

	np.save(direc_run+"preds_averaged.npy", test_2)
	np.save(direc_run+"preds.npy", test_results)

	if X_preds is not None:
		preds = prediction_averaging_sequential(files, X_preds)[:,1]
		np.save(direc_run+"full_preds.npy", preds)
	else: 
		preds=None

	if samples is not None:
		samples_preds = prediction_averaging_sequential(files, samples)
		np.save(direc_run+"samples_preds.npy", samples_preds)
	
	return model, files, results, preds
	


