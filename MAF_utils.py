import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.stats import gaussian_kde
import classifier_utils as cl
import yaml
import os
import tensorflow as tf
import tensorflow_probability as tfp
import plotting_utils as pf
import dataprep_utils as dp
from keras.utils.np_utils import to_categorical 

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors


def make_degrees(p, hidden_dims):
    m = [tf.constant(range(1, p + 1 ))]
    for dim in hidden_dims:
        n_min = min(np.min(m[-1]), p - 1)
        degrees = (np.arange(dim) % max(1, p - 1) + min(1, p - 1))
        degrees = tf.constant(degrees, dtype="int32")
        m.append(degrees)
    return m

def make_masks(degrees):
    masks = [None] * len(degrees)
    for i, (ind, outd) in enumerate(zip(degrees[:-1], degrees[1:])):
        masks[i] = tf.cast(ind[:, tf.newaxis] <= outd, dtype="float32")        
    masks[-1] = tf.cast(degrees[-1][:, np.newaxis] < degrees[0][1:], dtype="float32")
    return masks

def make_constraint(mask):    
    def _constraint(x):
        #print(tf.shape(mask))
        #print(tf.identity(x))
        return mask * tf.identity(x)
    return _constraint

def make_init(mask):
    def _init(shape, dtype=None):
        return mask * tf.keras.initializers.GlorotUniform(23)(shape)
    return _init

def make_network(p, hidden_dims, params):
    masks = make_masks(make_degrees(p, hidden_dims))    
    masks[-1] = tf.tile(masks[-1][..., tf.newaxis], [1, 1, params])
    masks[-1] = tf.reshape(masks[-1], [masks[-1].shape[0], (p-1) * params])
    
    network =  tf.keras.Sequential([
        tf.keras.layers.InputLayer((p,))
    ])
    for dim, mask in zip(hidden_dims + [(p-1) * params], masks):
        layer = tf.keras.layers.Dense(
            dim,
            kernel_constraint=make_constraint(mask),
            #kernel_initializer=make_init(mask),
            activation=tf.nn.leaky_relu)
        network.add(layer) 
        #norm = tf.keras.layers.BatchNormalization()
        #network.add(norm)
    network.add(tf.keras.layers.Reshape([p-1, params]))
    
    return network


class MAF(tfb.Bijector):
    def __init__(self, shift_and_log_scale_fn, name="maf"):
        super(MAF, self).__init__(forward_min_event_ndims=1, name=name)
        self._shift_and_log_scale_fn = shift_and_log_scale_fn
        
    def _shift_and_log_scale(self, y):
        params = self._shift_and_log_scale_fn(y)          
        shift, log_scale = tf.unstack(params, num=2, axis=-1)
        return shift, log_scale
        
    def _forward(self, x):
        y = tf.zeros_like(x, dtype=tf.float32)
        for i in range(x.shape[-1]):            
            shift, log_scale = self._shift_and_log_scale(y)            
            y = tf.concat([x[:,:1],x[:,1:] * tf.math.exp(log_scale) + shift], axis=1)
            #y = tf.concat([x[:,0],y],axis=1)
        return y

    def _inverse(self, y):
        shift, log_scale = self._shift_and_log_scale(y)
        return tf.concat([y[:,:1],(y[:,1:] - shift) * tf.math.exp(-log_scale)],axis=1)

    def _inverse_log_det_jacobian(self, y):
        _, log_scale = self._shift_and_log_scale(y)
        return -tf.reduce_sum(log_scale, axis=self.forward_min_event_ndims)

def make_model_MAF(inputs, hidden_dim, num_layers, lr=1e-4, params=2):
	bijectors = []#[tfb.BatchNormalization()]
	for i in range(0, num_layers):
		made = make_network(inputs, hidden_dim, params)
		permute = np.append([0],np.random.permutation(range(1,inputs)))
		bijectors.append(MAF(made))
		bijectors.append(tfb.Permute(permutation=permute)) 
		
	bijectors = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))

	distribution = tfd.TransformedDistribution(
		distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(inputs)),
		bijector=bijectors
	)

	x_ = tf.keras.layers.Input(shape=(inputs,), dtype=tf.float32)
	log_prob_ = distribution.log_prob(x_)
	model = tf.keras.Model(x_, log_prob_)

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=lambda _, log_prob: -log_prob)
	return model, bijectors

def model_averaging_MAF(files, model, bijectors):

	weights = []
	for i,file_i in enumerate(files):
		model.load_weights(file_i)
		weights = np.append(weights, model.get_weights())
	weights = np.reshape(weights, (len(files), len(weights)//len(files)))

	new_weights = np.mean(weights, axis=0)
	model.set_weights(new_weights)
	return model, bijectors

def model_averaging_realnvp(model, files):

	weights = []
	for i,file_i in enumerate(files):
		model.load_weights(file_i)
		weights = np.append(weights, model.get_weights())
	weights = np.reshape(weights, (len(files), len(weights)//len(files)))

	new_weights = np.mean(weights, axis=0)
	model.set_weights(new_weights)
	return model


def sample_averaging_MAF(model, bijectors, files, m, N, inputs=4):
	N_each = int(N/len(files))
	
	for i,file_i in enumerate(files):
		model.load_weights(file_i)
		X_current = np.array(sample_MAF(bijectors, N_each, inputs, m))
		if i==0:
			X_s = X_current
		else:
			X_s = np.concatenate((X_s,X_current),axis=0)
	np.random.shuffle(X_s)
	return X_s

def sample_MAF(bijectors, N, inputs, m):
	gaussian_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(inputs-1))
	kernel = gaussian_kde(m)
	m_samples = kernel.resample(size=int(N)).T
	samples_gaussian = gaussian_dist.sample(N)
	samples = np.concatenate((m_samples, samples_gaussian), axis=1)
	samples = tf.cast(samples, dtype=tf.float32)
	samples = bijectors._forward(samples)
	return samples

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

def filter_for_finite(arr):
	filtering = np.ones(len(arr),dtype=bool)
	for i in range(arr.shape[-1]):
		filtering*= np.isfinite(arr[:,i])
	return arr[filtering]

def run_MAF(args, X_train, X_val, m_inner, m_outer, direc, normalisation, X_inner_test=None, X_outer_test=None):
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

	model, bijectors = make_model_MAF(args.inputs+1, hidden_dim=params['hidden_dim'], num_layers=int(params['num_layers']), lr=float(params['lr']), params=int(params['params']))

	history = model.fit(
		X_train, y = np.zeros((X_train.shape[0],0)), batch_size=int(params['batch_size']), epochs=int(params['epochs']), verbose=2, validation_data = (X_val,  np.zeros((X_val.shape[0],0))), callbacks =[model_checkpoint_callback]
	)

	np.save(direc+'DE_history.npy', history.history)

	files = cl.take_best(folder, history, best = args.DE_N_best_epochs)
	np.save(direc+"best_files_DE.npy", files)

	model.save_weights(direc+"MAF_model.h5")

	pf.plot_training(history,"training_density", directory=direc)

	if args.no_averaging:
		print()
		print("Sample without averaging")
		samples_inner = sample_MAF(bijectors, args.N_samples, args.inputs+1, m_inner)
		samples_inner = filter_for_finite(normalisation.inverse(samples_inner))
		np.save(direc+"samples_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer = sample_MAF(bijectors, args.N_samples, args.inputs+1, m_outer)
		samples_outer = filter_for_finite(normalisation.inverse(samples_outer))
		np.save(direc+"samples_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	if args.ensemble:
		print()
		print("Sample with ensemble")
		samples_inner = sample_averaging_MAF(model, bijectors, files, m_inner, args.N_samples, inputs=args.inputs+1)
		samples_inner = filter_for_finite(normalisation.inverse(samples_inner))
		np.save(direc+"samples_ens_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_ens_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer = sample_averaging_MAF(model, bijectors, files, m_outer, args.N_samples, inputs=args.inputs+1)
		samples_outer = filter_for_finite(normalisation.inverse(samples_outer))
		np.save(direc+"samples_ens_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_ens_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	if args.weight_averaging:
		print()
		print("Sample with weight averaging")
		new_model, new_bijectors = model_averaging_MAF(files, model, bijectors)
		new_model.save_weights(direc+"MAF_avg_model.h5")
	
		samples_inner = sample_MAF(new_bijectors, args.N_samples, args.inputs+1, m_inner)
		samples_inner = filter_for_finite(normalisation.inverse(samples_inner))
		np.save(direc+"samples_avg_inner.npy", samples_inner)
		if X_inner_test is not None and args.DE_test_on_inner:
			pf.densities_plot_two(X_inner_test, samples_inner, title=direc+"samples_avg_inner", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_inner_test, samples_inner, args, direc)

		samples_outer = sample_MAF(new_bijectors, args.N_samples, args.inputs+1, m_outer)
		samples_outer = filter_for_finite(normalisation.inverse(samples_outer))
		np.save(direc+"samples_avg_outer.npy", samples_outer)
		if X_outer_test is not None and args.DE_test_on_outer:
			pf.densities_plot_two(X_outer_test, samples_outer, title=direc+"samples_avg_outer", name1="Data", name2="Samples", logit=False)
			test_with_full_classifier(X_outer_test, samples_outer, args, direc)

	return samples_inner, samples_outer
