import numpy as np
import classifier_utils as cl
import MAF_utils as MAF
import RealNVP_utils as RNVP
import dataprep_utils as dp
import bumphunt_utils as bh
import plotting_utils as pf
import hyperparameter_utils as hp
import argparse
import os


parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised"], required=True)
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

#HP Arguments
parser.add_argument('--hp_all', default=False, action="store_true")
parser.add_argument('--hp_learning_rate', default=False, action="store_true")
parser.add_argument('--hp_label_smoothing', default=False, action="store_true")
parser.add_argument('--hp_dropout', default=False, action="store_true")
parser.add_argument('--hp_batchsize', default=False, action="store_true")
parser.add_argument('--hp_l1', default=False, action="store_true")
parser.add_argument('--hp_l2', default=False, action="store_true")
parser.add_argument('--hp_beta1', default=False, action="store_true")

args = parser.parse_args()

if args.hp_all:
	args.hp_learning_rate = True 
	args.hp_label_smoothing = True 
	args.hp_dropout = True 
	args.hp_batchsize = True 
	args.hp_l1 = True 
	args.hp_l2 = True 
	args.hp_beta1 = True 

if args.cl_filename is None:
	if args.gaussian_inputs:
		args.cl_filename = "classifier"+str(args.N_normal_inputs)+".yml"
	else:
		args.cl_filename = "classifier"+str(args.inputs)+".yml"

if not args.ensemble and not args.weight_averaging and not args.no_averaging and args.mode=="cathode" and args.samples_file is None:
	raise ValueError("Need at least one method of sample generation if running cathode without sample file")

if not os.path.exists(args.directory):
		os.makedirs(args.directory)

print("args:", args)
print()

print("Running in mode:", args.mode)

if args.mode in ["IAD", "supervised"] and (args.minmass != 3.3 or args.maxmass != 3.7):
	raise ValueError("IAD and supervised incompatible with different signal regions")

if args.mode in ["cwola", "IAD", "supervised"]:
	X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, _, _ = dp.classifier_data_prep(args)
	direc_run = args.directory
	hp.hyperparameter_optimization(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, direc_run=direc_run)

elif args.mode == "cathode" and not args.DE_train_every_run:
	if args.samples_file is None:
		raise ValueError("Hyperparameter optimization requires samples to be given")
	else:
		X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, norm, cl_samples = dp.classifier_data_prep(args)
	direc_run = args.directory
	hp.hyperparameter_optimization(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, direc_run=direc_run, samples=cl_samples)

else:
	raise ValueError("No valid mode chosen")

	

