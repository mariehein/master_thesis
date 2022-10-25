import numpy as np
import classifier_utils as cl
import MAF_utils as MAF
import RealNVP_utils as RNVP
import dataprep_utils as dp
import plotting_utils as pf
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
parser.add_argument('--supervised_normal_signal', default=False, action="store_true")
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
parser.add_argument('--cathode_train_on_outer', default=False, action="store_true")


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

args = parser.parse_args()

if args.cl_filename is None:
	if args.gaussian_inputs:
		args.cl_filename = "classifier"+str(args.N_normal_inputs)+".yml"
	else:
		args.cl_filename = "classifier"+str(args.inputs)+".yml"
	
if args.DE_filename is None:
	args.DE_filename = str(args.DE_type)+".yml"

if not args.ensemble and not args.weight_averaging and not args.no_averaging and args.mode=="cathode" and args.samples_file is None:
	raise ValueError("Need at least one method of sample generation if running cathode without sample file")

if not os.path.exists(args.directory):
		os.makedirs(args.directory)

print("args:", args)
print()

print("Running in mode:", args.mode)

if args.mode in ["IAD", "supervised"] and (args.minmass != 3.3 or args.maxmass != 3.7):
	raise ValueError("IAD and supervised incompatible with different signal regions")

if args.cathode_train_on_outer and (args.mode!="cathode" or args.samples_file is None or not args.k_fold):
	raise ValueError("CATHODE train on outer not compatible wiht other options")

if args.k_fold:
	if args.mode =="cwola" or (args.mode =="cathode" and not args.DE_train_every_run and args.samples_file is not None):
		for i in range(5):
			print()
			print("------------------------------------------------------")
			print()
			print("Classifier run no. ", i)
			print()
			direc_run = args.directory+"run"+str(i)+"/"
			X_train, Y_train, X_val, Y_val, X_test, Y_test, normalisation, samples_test = dp.k_fold_data_prep(args, i, direc_run)
			cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, direc_run=direc_run, samples=samples_test)

	elif args.mode =="cathode" and not args.DE_train_every_run:
		X_train, X_val, m_inner, m_outer, X_inner_test, X_outer_test, normalisation = dp.DE_data_prep(args)
		if args.DE_type=="RealNVP":
			samples, _ = RNVP.run_realnvp(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=X_outer_test)
		elif args.DE_type=="MAF":
			samples, _ = MAF.run_MAF(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=X_outer_test)
		for i in range(5):
			print()
			print("------------------------------------------------------")
			print()
			print("Classifier run no. ", i)
			print()
			direc_run = args.directory+"run"+str(i)+"/"
			X_train, Y_train, X_val, Y_val, X_test, Y_test, normalisation, samples_test = dp.k_fold_data_prep(args, i, direc_run, samples)
			cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, direc_run=direc_run, samples=samples_test)
	else: 
		raise ValueError("Problem found: mode not compatible with k fold validation or invalid CATHODE option")
	
elif args.mode in ["cwola", "IAD", "supervised"]:
	X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, _, _ = dp.classifier_data_prep(args)
	for i in range(args.start_at_run, args.N_runs):
		print()
		print("------------------------------------------------------")
		print()
		print("Classifier run no. ", i)
		print()
		direc_run = args.directory+"run"+str(i)+"/"
		model, files, results, preds = cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, direc_run=direc_run)

elif args.mode == "cathode" and not args.DE_train_every_run:
	if args.samples_file is None:
		X_train, X_val, m_inner, m_outer, X_inner_test, X_outer_test, normalisation = dp.DE_data_prep(args)
		if args.DE_type=="RealNVP":
			samples, _ = RNVP.run_realnvp(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=X_outer_test)
		elif args.DE_type=="MAF":
			samples, _ = MAF.run_MAF(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=X_outer_test)
		X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, norm, cl_samples = dp.classifier_data_prep(args, samples)
	else:
		X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, norm, cl_samples = dp.classifier_data_prep(args)
	for i in range(args.start_at_run, args.N_runs):
		print()
		print("------------------------------------------------------")
		print()
		print("Classifier run no. ", i)
		print()
		direc_run = args.directory+"run"+str(i)+"/"
		model, files, results, preds = cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, direc_run=direc_run, samples=cl_samples)

elif args.mode == "cathode" and args.DE_train_every_run:
	for i in range(args.start_at_run, args.N_runs):
		print()
		print("------------------------------------------------------")
		print()
		print("DE run no. ", i)
		print()
		direc_run = args.directory+"run"+str(i)+"/"
		X_train, X_val, m_inner, m_outer, X_inner_test, X_outer_test, normalisation = dp.DE_data_prep(args)
		if args.DE_type=="RealNVP":
			samples, _ = RNVP.run_realnvp(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=None)
		elif args.DE_type=="MAF":
			samples, _ = MAF.run_MAF(args, X_train, X_val, m_inner, m_outer, args.directory, normalisation, X_inner_test=X_inner_test, X_outer_test=None)
		X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, label_preds, m_preds, norm, cl_samples = dp.classifier_data_prep(args, samples)

		print()
		print("------------------------------------------------------")
		print()
		print("Classifier run no. ", i)
		print()
		model, files, results, preds = cl.classifier_training(args, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_preds, direc_run=direc_run, samples=cl_samples)
else:
	raise ValueError("No valid mode chosen")

	

