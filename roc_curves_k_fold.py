import numpy as np
import classifier_utils as cl
import MAF_utils as MAF
import RealNVP_utils as RNVP
import dataprep_utils as dp
import bumphunt_utils as bh
import plotting_utils as pf
import argparse
import os


parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised"], required=True) #choose cwola here
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--cl_filename', type=str, default=None)
parser.add_argument('--DE_filename', type=str, default=None)

#Data Preparation Arguments
parser.add_argument('--data_file', type=str, default="/home/zu992399/master_thesis/not_crazy/jet/events_anomalydetection_v2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/zu992399/images_data/data/events_anomalydetection_qcd_extra_inneronly_features.h5")
parser.add_argument('--signal_percentage', type=float, default=None)
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--N_train', type=int, default=None)
parser.add_argument('--N_val', type=int, default=None)
parser.add_argument('--N_test', type=int, default=200000)
parser.add_argument('--ssb_width', type=float, default=0.2)
parser.add_argument('--gaussian_distortion', type=float, default=None)
parser.add_argument('--cl_logit', default=False, action="store_true")
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--DE_logit', default=True, action="store_false")
parser.add_argument('--DE_norm', default=True, action="store_false")
parser.add_argument('--test_on_input', default=False, action="store_true")
parser.add_argument('--gaussian_inputs', default=False, action="store_true")
parser.add_argument('--N_normal_inputs', default=4, type=int)

#Classifier Arguments
parser.add_argument('--use_class_weights', default=True, action="store_false")
parser.add_argument('--use_val_weights', default=True, action="store_false")
parser.add_argument('--N_runs', type=int, default=10)
parser.add_argument('--start_at_run', type=int, default=0)
parser.add_argument('--cl_N_best_epochs', type=int, default=10)
parser.add_argument('--cl_detailed_plots', default=False, action="store_true")
parser.add_argument('--cl_loss_tracking', default=False, action="store_true")
parser.add_argument('--k_fold', default=False, action="store_true")

#DE Arguments
parser.add_argument('--DE_type', default="RealNVP", type=str, choices=["MAF", "RealNVP"])
parser.add_argument('--DE_train_every_run', default=False, action="store_true")
parser.add_argument('--samples_file', type=str, default=None)
parser.add_argument('--ensemble', default=False, action="store_true")
parser.add_argument('--weight_averaging', default=False, action="store_true")
parser.add_argument('--no_averaging', default=False, action="store_true")
parser.add_argument('--N_samples', type=int, default=1e6)
parser.add_argument('--DE_N_best_epochs', type=int, default=10)
parser.add_argument('--DE_test_on_inner', default=True, action="store_false")
parser.add_argument('--DE_test_on_outer', default=True, action="store_false")
parser.add_argument('--DE_test_independently', default=False, action="store_true")

#Bump Hunt Arguments
parser.add_argument('--BH_percentiles', default=[], type=float, nargs='*')
parser.add_argument('--fit_width', type=float, default=0.4)
parser.add_argument('--fit_min', type=float, default=None)
parser.add_argument('--fit_max', type=float, default=None)
parser.add_argument('--fit_binwidth', type=float, default=0.1)

args = parser.parse_args()

if args.cl_filename is None:
	if args.gaussian_inputs:
		args.cl_filename = "classifier"+str(args.N_normal_inputs)+".yml"
	else:
		args.cl_filename = "classifier"+str(args.inputs)+".yml"
	
if args.DE_filename is None:
	args.DE_filename = str(args.DE_type)+".yml"

if args.fit_min is None:
	args.fit_min = args.minmass-args.fit_width

if args.fit_max is None:
	args.fit_max = args.maxmass+args.fit_width

if not os.path.exists(args.directory):
		os.makedirs(args.directory)

print("args:", args)
print()

print("Running in mode:", args.mode)

X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, _, _ = dp.classifier_data_prep(args)

folder = args.directory #"results/k_fold/cathode_sliding/window5/"
name = "test_avg"

for i in range(5):
	files = np.load(folder+"run"+str(i)+"/best_files_classifier.npy")
	preds = cl.prediction_averaging_sequential(files, X_test)
	pf.plot_roc(preds[:,1], Y_test[:,1],title="roc_"+name, directory=folder, direc_run=folder+"run"+str(i)+"/")


	

