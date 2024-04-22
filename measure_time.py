from train_test_api.utils import *
from test_algorithm.TestAlgorithm_for_csv import *
from genetic_algorithm.parallelise_to_csv import *
import pandas as pd
import os
import glob
import time

min_date_eval = "2021-03-01"
output_path = "output/"
data_path="data_obfuscated_time/"
lsTraining = [365]

with open("data/allfeatures", "r") as fp:
            features = json.load(fp)

ls_scenari = {"enet_pred_RS" : "results/final_prediction/hyperparameter/hp_enet_pred_RS/enet_pred_RS_2560767.csv",
"xgb_pred_RS" : "results/final_prediction/hyperparameter/hp_xgb_pred_RS/xgb_pred_RS_2561308.csv",
"GeneticSingleIs_GA" : "results/final_prediction/hyperparameter/hp_GeneticSingleIs_GA/GeneticSingleIs_GA_2559453.csv",
"GeneticSingleIs_GA_1000" : "results/final_prediction/hyperparameter/hp_GeneticSingleIs_GA/GeneticSingleIs_GA_2559453.csv"}

results_data = []

for scenario, file_i in ls_scenari.items():
    # get the date
    output_folder = output_path + scenario + "/test/" + min_date_eval + "/"

    # Get the params files
    nb_esn = 1
    if scenario in ["GeneticSingleIs_GA", "GeneticSingleIs_GA_PCA", "GeneticSingleIs_RS", "SingleIs_GA", "SingleIs_RS", "GeneticSingleIs_GA_10esn", "GeneticSingleIs_GA_20esn", "GeneticSingleIs_GA_20esn_week"]:
        nb_best_trials = 40
        units = 500
    elif scenario in ["xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS"]:
        nb_best_trials = 1
        units = 500
    elif scenario in ["GeneticSingleIs_GA_1000"]:
        nb_best_trials = 40
        units = 2000

    # Get all the trials as a DataFrame and remove trials with missing values
    trials_df = pd.read_csv(file_i, on_bad_lines = "skip").dropna()
    
    # Sort the trials based on the objective value in descending order and get top trials
    top_10_trials = trials_df.sort_values('value', ascending=False).tail(nb_best_trials)
    trial_ids = top_10_trials['job_id'].tolist()
    # make a big list with anteriority
    full_list = [(mintraining, trial_id) for mintraining in lsTraining for trial_id in trial_ids]
    
    start_time = time.time()
    # evaluate trials
    for meta in full_list:
    	trial_id = meta[1]
    	mintraining = meta[0]
    	job_id = "trial_"+str(trial_id)+"_train"+str(mintraining)
    	# get params
    	params = trials_df[trials_df["job_id"] == trial_id].to_dict(orient="records")[0]
    	
    	if scenario == "GeneticSingleIs_GA_1000" :
    	    params["ridge"] = 1e3
    	
    	temp = params.pop("value")
    	temp = params.pop("job_id")
    	# get trial
    	eval_objective_function(
    	  units = units,
    	  rm_output_files=False,
    	  params=params,
    	  features=features,
    	  output_path=output_folder,
    	  data_path=data_path,
    	  job_id=job_id,
    	  nb_esn = nb_esn,
    	  is_training=False,
    	  min_date_eval=min_date_eval
    	  )

    end_time = time.time()

    elapsed_time = end_time - start_time

    results_data.append({
        'Scenario': scenario,
        'Elapsed Time': elapsed_time
    })

# Create a DataFrame from the results_data list
results_df = pd.DataFrame(results_data)

results_df.to_csv("results/timing.csv")
