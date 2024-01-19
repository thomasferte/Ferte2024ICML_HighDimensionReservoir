from train_test_api.utils import *
from genetic_algorithm.parallelise_to_csv import *
import os
import glob
import random

array_id = os.getenv('SLURM_ARRAY_TASK_ID')

with open("data/allfeatures", "r") as fp:
            features = json.load(fp)

output_path = "/beegfs/tferte/output/2000_units/test/"
data_path = "data_obfuscated/"
study_path = "/beegfs/tferte/output/final_predictions/GeneticSingleIs_GA/GeneticSingleIs_GA_2559453.csv"

# Get all the trials as a DataFrame and remove trials with missing values
trials_df = pd.read_csv(study_path, on_bad_lines = "skip").dropna()

# Sort the trials based on the objective value in descending order and get top trials
top_40_trials = trials_df.sort_values('value', ascending=False).tail(40)
trial_ids = top_40_trials['job_id'].tolist()

# randomly sample trial_id
trial_id = trial_ids[int(array_id)]
params = trials_df[trials_df["job_id"] == trial_id].to_dict(orient="records")[0]
# randomly sample ridge
ridge = 1e3
params["ridge"] = ridge

job_id = trial_id + "_ridge_" + str(round(ridge))

eval_objective_function(
	  units = 2000,
	  rm_output_files=False,
	  params=params,
	  features=features,
	  output_path=output_path,
	  data_path=data_path,
	  job_id=job_id,
	  nb_esn = 1,
	  is_training=False,
	  min_date_eval='2021-03-01'
	  )
