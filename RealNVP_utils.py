import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from scipy.stats import gaussian_kde
import dataprep_utils as dp
from keras.utils.np_utils import to_categorical 
import plotting_utils as pf
import classifier_utils as cl
import yaml
import os

models = keras.models
layers = keras.layers
regularizers = keras.regularizers

output_dim = 256
reg = 0.01

def Coupling(input_shape, num_hidden=0):
    input = keras.layers.Input(shape=input_shape + 1)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    for i in range(num_hidden):
        t_layer_1 = keras.layers.Dense(
            input_shape, activation="relu", kernel_regularizer=regularizers.l2(reg)
        )(t_layer_1)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)

    initializer = keras.initializers.Zeros()
    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    for i in range(num_hidden):
        s_layer_1 = keras.layers.Dense(
            input_shape, activation="relu", kernel_regularizer=regularizers.l2(reg)
        )(s_layer_1)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg), kernel_initializer=initializer
    )(s_layer_1)
    return keras.Model(inputs=input, outputs=[t_layer_5, s_layer_5])


class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers, num_hidden=0, dynamic=True, inputs=4, masks=None):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.]*inputs, scale_diag=[1.]*inputs
        )
        
        if masks is None:
            self.masks =tf.convert_to_tensor(np.random.randint(2, size=(num_coupling_layers,inputs)),dtype=tf.float32)
        else:
            self.masks = masks
        print(self.masks.shape)
        #self.masks = np.array(
        #    [[0, 0, 1, 1], [1, 1, 0, 0],[0, 1, 1, 0], [1, 0, 0, 1],[0, 1, 1, 1], [1, 1, 1, 0],[0, 1, 0, 1], [1, 0, 1, 0]] * (num_coupling_layers // 8), dtype="float32"
        #)
        self.num_hidden = num_hidden
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(inputs, num_hidden=self.num_hidden) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    @tf.function  # autograph.experimental.do_not_convert
    def call(self, x, training=True, direction=None):
        m = x[:, :1]
        x = x[:, 1:]
        log_det_inv = 0
        if direction is None:
            direction = 1
            if training:
                direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](tf.concat([m, x_masked], axis=1))
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return tf.concat([m, x], axis=1), log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y[:, 1:]) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

def sample_from_model(model, m, N, kernel=None):
	if kernel is None:	
		kernel = gaussian_kde(m)

	samples_gaussian = model.distribution.sample(N)
	m_samples = kernel.resample(size=int(N))  # sample with kde
	samples, _ = model.predict(np.concatenate((m_samples.T, samples_gaussian), axis=1))
	return samples, kernel
"""
def model_averaging_realnvp(files, X, masks=None, num_coupling=16, num_hidden=1, inputs=4):
	models = []
	
	for i,file_i in enumerate(files):
		models = np.append(models, RealNVP(num_coupling_layers = num_coupling, num_hidden = num_hidden,inputs=inputs, masks=masks))
		models[i].compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))	
		models[i].fit(X, epochs=1, verbose=2,validation_split=0.2)	
		models[i].load_weights(file_i)

	model = RealNVP(num_coupling_layers = num_coupling, num_hidden = num_hidden, inputs=inputs, masks=masks)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

	history = model.fit(
		X, batch_size=1048, epochs=1, verbose=2, validation_split=0.2
	)

	for i in range(num_coupling):
		new_weights = list()
		weights = [model.layers_list[i].get_weights() for model in models]
		for weights_list_tuple in zip(*weights):
			new_weights.append(np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))
		model.layers_list[i].set_weights(new_weights)

	return model
"""

def model_averaging_realnvp(model, files):

	weights = []
	for i,file_i in enumerate(files):
		model.load_weights(file_i)
		weights = np.append(weights, model.get_weights())
	weights = np.reshape(weights, (len(files), len(weights)//len(files)))

	#for weights_list_tuple in zip(*weights)
	#	new_weights.append(np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))
	new_weights = np.mean(weights, axis=0)
	model.set_weights(new_weights)
	return model

def sample_averaging_realnvp(model, files, m, N):
	N_each = int(N/len(files))
	
	for i,file_i in enumerate(files):
		model.load_weights(file_i)
		X_current, _ = sample_from_model(model, m, N_each)
		if i==0:
			X= X_current
		else:
			X = np.concatenate((X,X_current),axis=0)
	np.random.shuffle(X)
	return X

def test_with_full_classifier(X_test, samples, args, direc):
	X = np.concatenate((X_test, samples[:len(X_test)]), axis=0)[:,1:]
	if args.cl_norm:
		if args.cl_logit:
			norm = dp.logit_norm(X)
		else:
			norm = dp.no_logit_norm(X)
		X, _ = norm.forward(X)
	Y = to_categorical(np.concatenate((np.ones(len(X_test)),np.zeros(len(X)-len(X_test))),axis=0))
	X, Y = dp.shuffle_XY(X, Y)
	X_train, X_val, X_test = np.array_split(X, 3)
	Y_train, Y_val, Y_test = np.array_split(Y, 3)
	cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, direc_run=direc, DE=True)

def run_realnvp(args, X_train, X_val, m_inner, m_outer, direc, normalisation, X_inner_test=None, X_outer_test=None):
	with open(args.DE_filename, 'r') as stream:
		params = yaml.safe_load(stream)	

	if not os.path.exists(direc):
		os.makedirs(direc)

	folder = direc+"DE_models/"	
	if not os.path.exists(folder):
		os.makedirs(folder)

	checkpoint_filepath = folder+'{epoch:02d}.hdf5'
	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		monitor='val_loss',)	

	model = RealNVP(num_coupling_layers = int(params['num_coupling']), num_hidden = int(params['num_hidden']), inputs=args.inputs)

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(params['lr'])))

	history = model.fit(
		X_train, batch_size=int(params['batch_size']), epochs=int(params['epochs']), verbose=2, validation_data=(X_val, ), callbacks =[model_checkpoint_callback]
	)

	np.save(direc+'DE_history.npy', history.history)

	files = cl.take_best(folder, history, best = args.DE_N_best_epochs)
	np.save(direc+"best_files_DE.npy", files)
	np.save(direc+"DE_masks", model.masks)
	model.save_weights(direc+"RealNVP_model.h5")

	pf.plot_training(history,"training_density", directory=direc)

	if args.no_averaging:
		print()
		print("Sample without averaging")
		samples_inner, _ = sample_from_model(model, m_inner, args.N_samples)
		samples_inner = normalisation.inverse(samples_inner)
		np.save(direc+"samples_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer, _ = sample_from_model(model, m_outer, args.N_samples)
		samples_outer = normalisation.inverse(samples_outer)
		np.save(direc+"samples_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	if args.ensemble:
		print()
		print("Sample with ensemble")
		samples_inner = sample_averaging_realnvp(model, files, m_inner, args.N_samples)
		samples_inner = normalisation.inverse(samples_inner)
		np.save(direc+"samples_ens_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_ens_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer = sample_averaging_realnvp(model, files, m_outer, args.N_samples)
		samples_outer = normalisation.inverse(samples_outer)
		np.save(direc+"samples_ens_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_ens_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	if args.weight_averaging:
		print()
		print("Sample with weight averaging")
		new_model = model_averaging_realnvp(model, files)
		new_model.save_weights(direc+"RealNVP_avg_model.h5")
	
		samples_inner, _ = sample_from_model(new_model, m_inner, args.N_samples)
		samples_inner = normalisation.inverse(samples_inner)
		np.save(direc+"samples_avg_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_avg_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer, _ = sample_from_model(new_model, m_outer, args.N_samples)
		samples_outer = normalisation.inverse(samples_outer)
		np.save(direc+"samples_avg_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_avg_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	return samples_inner, samples_outer


	





