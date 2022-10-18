import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotting_utils as pf
import yaml
from keras.utils.np_utils import to_categorical 
import classifier_utils as cl
import keras_tuner as kt

models = keras.models
layers = keras.layers
regularizers = keras.regularizers

def hyperparameter_optimization(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds=None, direc_run=None, DE=False, samples=None):
	if direc_run is None:	
		direc_run=args.directory
	
	with open(args.cl_filename, 'r') as stream:
		params = yaml.safe_load(stream)

	class MyModel(kt.HyperModel):
		def build(self, hp):
			if args.hp_learning_rate:
				self.lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
			else: 
				self.lr = float(params['lr'])
	
			if args.hp_label_smoothing:
				self.label_smoothing = hp.Float("label_smoothing", min_value=0, max_value=0.4)
			else: 
				self.label_smoothing = float(params['label_smoothing'])
	
			if args.hp_dropout:
				self.dropout = hp.Float("dropout", min_value=0, max_value=0.5)
			else: 
				self.dropout = float(params['dropout'])
	
			if args.hp_batchsize:
				exponent = hp.Int("batchsize", min_value=6, max_value=12)
				self.batchsize = 2**exponent
			else: 
				self.batchsize = int(params['batchsize'])	

			if args.hp_l1:
				self.l1 = hp.Float("l1", min_value=1e-10, max_value=1e-1, sampling="log")
			else: 
				self.l1 = float(params['l1'])

			if args.hp_l2:
				self.l2 = hp.Float("l2", min_value=1e-10, max_value=1e-1, sampling="log")
			else: 
				self.l2 = float(params['l2'])

			if args.hp_beta1:
				self.beta_1 = hp.Float("beta_1", min_value=0.5, max_value=0.99)
			else: 
				self.beta_1 = float(params['beta_1'])	

			model = cl.make_model(hidden=3, lr=self.lr, momentum = self.beta_1, activation="relu", dropout=self.dropout, l1 = self.l1, l2 = self.l2, inputs = args.inputs, label_smoothing=self.label_smoothing)
			return model

		def fit(self, hp, model, *args,**kwargs):
			return model.fit(
				*args,
				batch_size=self.batchsize,
				shuffle=True,
				verbose=2,
				**kwargs
			)

	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

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

	tuner = kt.Hyperband(MyModel(), objective = kt.Objective('val_loss','min'), directory = args.directory, max_epochs=30, hyperband_iterations = 1, project_name="optim")

	tuner.search(X_train, Y_train, validation_data = (X_val,Y_val,val_sample_weights), class_weight = class_weight)

	tuner.results_summary()

	#print(tuner.get_best_hyperparameters(1)[0])
	best_model = tuner.get_best_models(1)[0]

	best_model.save(args.directory+"IAD.h5")

	# =============================================================================
	# ROC curve
	# =============================================================================

	test_results = best_model.predict(X_test).T[1]

	pf.plot_scores(test_results, Y_test[:,1], title="samples",directory=args.directory)
	pf.plot_roc(test_results, Y_test[:,1],title="roc_IAD",directory=args.directory)
	np.save(direc_run+"preds.npy", test_results)
	


