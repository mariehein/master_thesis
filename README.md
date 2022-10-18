# Using Density Estimation for Resonance Searches at the LHC

This repository contains the code used to produce the results of my master thesis with the above title, in which the CATHODE method was studied. 

*"Classifying Anomalies THrough Outer Density Estimation (CATHODE)"*,  
By Anna Hallin, Joshua Isaacson, Gregor Kasieczka, Claudius Krause, Benjamin Nachman,
Tobias Quadfasel, Matthias Schlaffer, David Shih, and Manuel Sommerhalder. <br>
[arXiv:2109.00546](https://arxiv.org/abs/2109.00546). 

## Data 

The data needed to run the code can be obtained through 
```
wget https://zenodo.org/record/4536377/files/events_anomalydetection_v2.features.h5
wget https://zenodo.org/record/5759087/files/events_anomalydetection_qcd_extra_inneronly_features.h5
```

## Run default setup

The default setup can be run with `run_pipeline.py`, which requires a mode and a directory in every case. For CATHODE a density estimator sampling method also needs to be specified. All four methods in their default configuration can be run as shown below
 
```
python run_pipeline.py --mode "supervised" --directory "give/supervised_directory/"
python run_pipeline.py --mode "IAD" --directory "give/IAD_directory/"
python run_pipeline.py --mode "cathode" --directory "give/CATHODE_directory/" --weight_averaging
python run_pipeline.py --mode "cwola" --directory "give/CWoLa_directory/"
```

## Visualization

Results can be visualized using `plotting.py` and `plotting_SIC.py`, which plot ROC and SIC curves respectively as medians with error band.

## Hyperparameter optimization

Hyperparameter optimization is performed by running `run_hyperparameter_optimization.py`. An optimization of all hyperparameters, which can be optimized with this program, can be performed for the IAD with

```
python run_hyperparameter_optimization.py --mode "IAD" --directory "give/hp_directory" --hp_all
```

## Bump Hunt

To perform the bump hunt, one run needs to be performed for all signal region windows with `--N_runs 1` for the full data configuration or once per signal region window with `--k_fold` for the k-fold cross validation method. The respective directory is then specified in either `bh_full_data.py` or `bh_k_fold.py`. Note that the option `cwola` needs to be set true or false depending on what method is used. The bump hunt can only be performed with CWoLa or CATHODE in the current configuration.
