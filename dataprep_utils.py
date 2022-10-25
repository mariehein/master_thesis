import numpy as np
from scipy.special import logit, expit
from keras.utils.np_utils import to_categorical 
import pandas as pd
import warnings
import plotting_utils as pf
import os

def shuffle_XY(X,Y):
    seed_int=np.random.randint(300)
    np.random.seed(seed_int)
    np.random.shuffle(X)
    np.random.seed(seed_int)
    np.random.shuffle(Y)
    return X,Y

class no_logit_norm:
	def __init__(self,array):
		self.mean = np.mean(array, axis=0)
		self.std = np.std(array, axis=0)

	def forward(self,array0):
		return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0),dtype=bool)

	def inverse(self,array0):
		return np.copy(array0)*self.std+self.mean

class logit_norm:
    def __init__(self, array0, mean=True):
        array = np.copy(array0)
        self.shift = np.min(array, axis=0)
        self.num = len(self.shift)
        self.max = np.max(array, axis=0) + self.shift
        if mean:
            finite=np.ones(len(array),dtype=bool)
            for i in range(self.num):
                array[:, i] = logit((array[:, i] + self.shift[i]) / self.max[i])
                finite *= np.isfinite(array[:, i])
            array=array[finite]
            self.mean = np.nanmean(array,axis=0)
            print(self.mean)
            self.std = np.nanstd(array,axis=0)
            print(self.std)
        self.do_mean = mean

    def forward(self, array0):
        array = np.copy(array0)
        finite = np.ones(len(array), dtype=bool)
        for i in range(self.num):
            array[:, i] = logit((array[:, i] + self.shift[i]) / self.max[i])
            if self.do_mean:
                array[:,i] = (array[:,i]-self.mean[i])/self.std[i]
            finite *= np.isfinite(array[:, i])
        return array[finite, :], finite

    def inverse(self, array0):
        array = np.copy(array0)
        for i in range(self.num):
            if self.do_mean:
                array[:,i] = array[:,i]*self.std[i] + self.mean[i]
            array[:,i] =expit(array[:,i])
            array[:, i] = array[:, i] * self.max[i] - self.shift[i]
        return array

def file_loading(filename, labels=True):
	if labels:
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'label']])
	else: 
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']])
		features = np.concatenate((features,np.zeros((len(features),1))),axis=1)
	E_part = np.sqrt(features[:,0]**2+features[:,1]**2+features[:,2]**2+features[:,3]**2)+np.sqrt(features[:,7]**2+features[:,8]**2+features[:,9]**2+features[:,10]**2)
	p_part = (features[:,0]+features[:,7])**2+(features[:,1]+features[:,8])**2+(features[:,2]+features[:,9])**2
	m_jj = np.sqrt(E_part**2-p_part)
	#p_t = np.array([np.max([np.sqrt(features[i,0]**2+features[i,1]**2), np.sqrt(features[i,7]**2+features[i,8]**2)]) for i in range(len(features))])
	ind=np.array(features[:,10]> features[:,3]).astype(int)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
		feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:,10]-features[:, 3])*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12] ,features[:,-1]])
		feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:,3]-features[:, 10])*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5] ,features[:,-1]])
	feat = feat1*ind+feat2*(np.ones(len(ind))-ind)
	feat = np.nan_to_num(feat)
	return feat.T

def Gaussian(ar, std):
	array = np.copy(ar)
	add = np.random.normal(scale=std, size=array.shape)
	return array + add

def Gaussian_distortion(X, Y, std):
	X[Y[:,1]==0]=Gaussian(X[Y[:,1]==0], std)
	return X




def classifier_data_prep(args, samples=None):
	print()

	data = file_loading(args.data_file)
	extra_bkg = file_loading(args.extrabkg_file, labels=False)
	
	sig = data[data[:,-1]==1]
	bkg = data[data[:,-1]==0]

	if args.signal_percentage is None:
		n_sig = 1000
	else:
		n_sig = int(args.signal_percentage*1000/0.6361658645922605)
	print("n_sig=", n_sig)
	
	data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
	np.random.seed(1)
	np.random.shuffle(data_all)
	extra_sig = sig[n_sig:]
	innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
	inner_extra_sig = extra_sig[innersig_mask]

	innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
	outermask = ~innermask
	innerdata = data_all[innermask]
	outerdata = data_all[outermask]

	extrabkg1 = extra_bkg[:312858]
	extrabkg2 = extra_bkg[312858:]

	if args.mode in ["IAD","supervised"]:
		print("idealized")
		if args.N_train is not None and args.N_val is not None:
			samples_train = extrabkg1[40000:40000+args.N_train]
			samples_val = extrabkg1[40000+args.N_train:40000+args.N_train+args.N_val]
		else:
			samples_train, samples_val = np.array_split(extrabkg1[40000:],2)
		print("N_train: ", len(samples_train), "; N_val: ", len(samples_val))
	elif args.mode=="cathode":
		print("cathode")
		if samples is None:
			samples = np.load(args.samples_file)
		if args.N_train is None:
			args.N_train = 200000
		if args.N_val is None:
			args.N_val = 200000
		samples_train = samples[:args.N_train]
		samples_val = samples[args.N_train:args.N_train+args.N_val]
		print("N_train: ", len(samples_train), "; N_val: ", len(samples_val))
	elif args.mode=="cwola":
		outer_data_ssb = outerdata[np.logical_and(outerdata[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > outerdata[:,0])]
		if args.N_train is not None and args.N_val is not None:
			samples_train = outer_data_ssb[:args.N_train]
			samples_val = outer_data_ssb[args.N_train:args.N_train+args.N_val]
		else:
			samples_train, samples_val = np.array_split(outer_data_ssb,2)
		print(np.min(outer_data_ssb[:,0]),np.max(outer_data_ssb[:,0]))
		print("N_train: ", len(samples_train), "; N_val: ", len(samples_val))
	else:
		raise ValueError("Invalid Mode")

	print(sum(innerdata[:60000,-1]))
	print(sum(innerdata[60000:120000,-1]))
	print(sum(outerdata[:500000,-1]))
	print(sum(outerdata[500000:,-1]))

	if args.mode=="supervised":
		if not args.supervised_normal_signal:
			sig_train, sig_val = np.array_split(inner_extra_sig[20000:],2)
		else: 
			sig_train = innerdata[:60000]
			sig_train = sig_train[sig_train[:,-1]==1]
			sig_val = innerdata[60000:120000]
			sig_val = sig_val[sig_val[:,-1]==1]
		X_train = np.concatenate((samples_train, sig_train), axis=0)
		Y_train = X_train[:,-1]
		if args.gaussian_inputs:
			gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
			X_train = np.concatenate((X_train[:,1:args.N_normal_inputs+1],gauss), axis=1)
		else:
			X_train = X_train[:,1:args.inputs+1]

		X_val = np.concatenate((samples_val, sig_val), axis=0)
		Y_val = X_val[:,-1]
		if args.gaussian_inputs:
			gauss = np.random.normal(size=(len(X_val),args.inputs-args.N_normal_inputs))
			X_val = np.concatenate((X_val[:,1:args.N_normal_inputs+1],gauss), axis=1)
		else:
			X_val = X_val[:,1:args.inputs+1]
	elif args.mode in ["IAD","cwola","cathode"]:
		if args.gaussian_inputs:
			X_train = np.concatenate((innerdata[:60000,1:args.N_normal_inputs+1],samples_train[:,1:args.N_normal_inputs+1]),axis=0)
			gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
			X_train = np.concatenate((X_train,gauss), axis=1)
			print(X_train.shape)
		else:
			X_train = np.concatenate((innerdata[:60000,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
		Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)		

		if args.gaussian_inputs:
			X_val = np.concatenate((innerdata[60000:120000,1:args.N_normal_inputs+1],samples_val[:,1:args.N_normal_inputs+1]),axis=0)
			gauss = np.random.normal(size=(len(X_val),args.inputs-args.N_normal_inputs))
			X_val = np.concatenate((X_val,gauss), axis=1)
		else:
			X_val = np.concatenate((innerdata[60000:120000,1:args.inputs+1],samples_val[:,1:args.inputs+1]),axis=0)
		Y_val = np.concatenate((np.ones(len(X_val)-len(samples_val)),np.zeros(len(samples_val))),axis=0)
	else: 
		raise ValueError('Wrong --args.mode given')

	if args.test_on_input:
		if args.mode in ["supervised","IAD"]:
			raise ValueError('supervised and IAD incompatible with test_on_input')
		elif args.mode=="cwola":
			X_test = np.concatenate((samples_val, extrabkg2[:args.N_test]))[:,1:args.inputs+1]
			Y_test = to_categorical(np.concatenate((np.zeros(len(samples_val)),np.ones(args.N_test))))
		elif args.mode=="cathode":
			X_test = np.concatenate((samples[N_train+N_val:N_train+N_val+N_test,1:args.inputs+1],extrabkg2[:N_test,1:args.inputs+1]))
			Y_test = to_categorical(np.concatenate((np.zeros(N_test),np.ones(N_test))))
	else:
		X_test = np.concatenate((extrabkg2,inner_extra_sig[:20000],extrabkg1[:40000]))
		Y_test = to_categorical(X_test[:,-1])
		if args.gaussian_inputs:
			gauss = np.random.normal(size=(len(X_test),args.inputs-args.N_normal_inputs))
			X_test = np.concatenate((X_test[:,1:args.N_normal_inputs+1],gauss), axis=1)
		else:
			X_test = X_test[:,1:args.inputs+1]

	X_train, Y_train = shuffle_XY(X_train, to_categorical(Y_train,2))
	X_val, Y_val = shuffle_XY(X_val, to_categorical(Y_val,2))

	if args.cl_logit:
		normalisation = logit_norm(X_train)
	else:
		normalisation = no_logit_norm(X_train)

	if args.cl_norm:
		X_train, _ = normalisation.forward(X_train)
		X_val, _ = normalisation.forward(X_val)
		X_test, _ = normalisation.forward(X_test)
		if args.gaussian_inputs:
			gauss = np.random.normal(size=(len(data_all),args.inputs-args.N_normal_inputs))
			X_preds = np.concatenate((data_all[:,1:args.N_normal_inputs+1],gauss), axis=1)
		else:
			X_preds = data_all[:,1:args.inputs+1]
		X_preds, finite_preds = normalisation.forward(X_preds)
		label_preds = data_all[finite_preds, -1]
		m_preds = data_all[finite_preds, 0]
		if samples is not None:
			samples, _ = normalisation.forward(samples[:,1:args.inputs+1])
	print("Train set: ", len(X_train), "; Val set: ", len(X_val), "; Test set: ", len(X_test))

	pf.densities_plot_two(X_train[Y_train[:,1]==1], X_train[Y_train[:,1]==0], args.directory+"samples_data.pdf", name1="data",
		                  name2="samples",)

	np.save(args.directory+"X_train.npy", X_train)
	np.save(args.directory+"X_val.npy", X_val)
	np.save(args.directory+"X_test.npy", X_test)
	np.save(args.directory+"Y_train.npy", Y_train)
	np.save(args.directory+"Y_val.npy", Y_val)
	np.save(args.directory+"Y_test.npy", Y_test)
	np.save(args.directory+"X_preds.npy", X_preds)
	np.save(args.directory+"m_preds.npy", m_preds)
	np.save(args.directory+"label_preds.npy", label_preds)

	if args.gaussian_distortion is not None:
		if args.mode=="IAD":
			X_train = Gaussian_distortion(X_train, Y_train, args.gaussian_distortion)
			X_val = Gaussian_distortion(X_val, Y_val, args.gaussian_distortion)
		else:
			raise ValueError('Gaussian Distortion only valid for IAD')

	return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, normalisation, samples

def k_fold_data_prep(args, k, direc_run, samples=None):
	print()
	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

	data = file_loading(args.data_file)
	extra_bkg = file_loading(args.extrabkg_file, labels=False)
	
	sig = data[data[:,-1]==1]
	bkg = data[data[:,-1]==0]

	if args.signal_percentage is None:
		n_sig = 1000
	else:
		n_sig = int(args.signal_percentage*1000/0.6361658645922605)
	print("n_sig=", n_sig)
	
	data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
	np.random.seed(1)
	np.random.shuffle(data_all)
	extra_sig = sig[n_sig:]
	innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
	inner_extra_sig = extra_sig[innersig_mask]

	innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
	outermask = ~innermask
	innerdata = data_all[innermask]
	outerdata = data_all[outermask]

	extrabkg1 = extra_bkg[:312858]
	extrabkg2 = extra_bkg[312858:]

	indices = np.roll(np.array(range(5)),k)
 	
	if args.mode=="cathode":
		print("cathode")
		if samples is None:
			samples = np.load(args.samples_file)
		if args.cathode_train_on_outer:
			samples = samples[np.logical_and(samples[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > samples[:,0])]
		if args.N_train is None:
			args.N_train = 160000
		if args.N_val is None:
			args.N_val = 160000
		samples_train = samples[:args.N_train]
		samples_val = samples[args.N_train:args.N_train+args.N_val]
		samples_test = samples[args.N_train+args.N_val:]
		print("N_train: ", len(samples_train), "; N_val: ", len(samples_val), "; N_test: ", len(samples_test))
	elif args.mode=="cwola":
		outer_data_ssb = outerdata[np.logical_and(outerdata[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > outerdata[:,0])]
		samples_t = np.array_split(outer_data_ssb,5)
		samples_train = np.concatenate((samples_t[indices[0]], samples_t[indices[1]]))
		samples_val = np.concatenate((samples_t[indices[2]], samples_t[indices[3]]))
		samples_test = samples_t[indices[4]]
		print("N_train: ", len(samples_train), "; N_val: ", len(samples_val), "; N_test: ", len(samples_test))
	else:
		raise ValueError("Invalid Mode for k fold")

	print(sum(innerdata[:60000,-1]))
	print(sum(innerdata[60000:120000,-1]))
	print(sum(outerdata[:500000,-1]))
	print(sum(outerdata[500000:,-1]))

	if args.cathode_train_on_outer:
		innerdata = outerdata[np.logical_and(outerdata[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > outerdata[:,0])]		

	if args.mode in ["cwola","cathode"]:
		X_t = np.array_split(innerdata,5)
		X_train = np.concatenate((X_t[indices[0]], X_t[indices[1]]))
		X_val = np.concatenate((X_t[indices[2]], X_t[indices[3]]))
		X_test = X_t[indices[4]]

		X_train = np.concatenate((X_train[:,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
		Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)		

		X_val = np.concatenate((X_val[:,1:args.inputs+1],samples_val[:,1:args.inputs+1]),axis=0)
		Y_val = np.concatenate((np.ones(len(X_val)-len(samples_val)),np.zeros(len(samples_val))),axis=0)

		Y_test = to_categorical(X_test[:,-1],2)
		X_test = X_test[:, 1:args.inputs+1]
		samples_test = samples_test[:,1:args.inputs+1]
	else: 
		raise ValueError('Wrong --args.mode given')

	X_train, Y_train = shuffle_XY(X_train, to_categorical(Y_train,2))
	X_val, Y_val = shuffle_XY(X_val, to_categorical(Y_val,2))

	if args.cl_logit:
		normalisation = logit_norm(X_train)
	else:
		normalisation = no_logit_norm(X_train)

	if args.cl_norm:
		X_train, _ = normalisation.forward(X_train)
		X_val, _ = normalisation.forward(X_val)
		X_test, _ = normalisation.forward(X_test)
		samples_test, _ = normalisation.forward(samples_test)
	print("Train set: ", len(X_train), "; Val set: ", len(X_val), "; Test set: ", len(X_test))

	pf.densities_plot_two(X_train[Y_train[:,1]==1], X_train[Y_train[:,1]==0], args.directory+"samples_data.pdf", name1="data",
		                  name2="samples",)

	np.save(direc_run+"X_train.npy", X_train)
	np.save(direc_run+"X_val.npy", X_val)
	np.save(direc_run+"X_test.npy", X_test)
	np.save(direc_run+"Y_train.npy", Y_train)
	np.save(direc_run+"Y_val.npy", Y_val)
	np.save(direc_run+"Y_test.npy", Y_test)
	np.save(direc_run+"samples_test.npy", samples_test)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test, normalisation, samples_test
	
def DE_data_prep(args):
	data = file_loading(args.data_file)
	extra_bkg = file_loading(args.extrabkg_file, labels=False)
	
	sig = data[data[:,-1]==1]
	bkg = data[data[:,-1]==0]

	if args.signal_percentage is None:
		n_sig = 1000
	else:
		n_sig = int(args.signal_percentage*1000/0.6361658645922605)
	
	data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
	np.random.seed(1)
	np.random.shuffle(data)
	extra_sig = sig[n_sig:]
	innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
	inner_extra_sig = extra_sig[innersig_mask]

	innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
	outermask = ~innermask
	innerdata = data_all[innermask]
	outerdata = data_all[outermask]

	extrabkg1 = extra_bkg[:312858]
	extrabkg2 = extra_bkg[312858:]

	X_train = outerdata[:500000,:args.inputs+1]
	X_val = outerdata[500000:,:args.inputs+1]
	X_inner_test = extrabkg1[:,:args.inputs+1]
	X_outer_test = X_val
	
	if args.DE_test_independently:
		X_val, X_outer_test = np.array_split(X_val, 2)

	if args.DE_logit:
		normalisation = logit_norm(X_train)
	else:
		normalisation = no_logit_norm(X_train)

	if args.DE_norm:
		X_train, _ = normalisation.forward(X_train)
		X_val, _ = normalisation.forward(X_val)



	print(len(X_train))
	print(len(X_val))
	
	m_inner = normalisation.forward(innerdata[:60000,:args.inputs+1])[0][:,0]
	m_outer = X_train[:,0]
	
	return X_train, X_val, m_inner, m_outer, X_inner_test, X_outer_test, normalisation

	
	
			 



	
		
	
	




