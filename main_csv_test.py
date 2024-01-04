from train_test_api.utils import *
from test_algorithm.TestAlgorithm_for_csv import *
from genetic_algorithm.parallelise_to_csv import *
import os
import glob

slurm_scenari = os.getenv('SLURM_JOB_NAME')
# slurm_scenari = "GeneticSingleIs_GA"
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
# array_id = 10

print(slurm_scenari + " " + str(array_id))

# Get the params files
nb_esn = 1
if slurm_scenari in ["GeneticSingleIs_GA", "GeneticSingleIs_RS", "SingleIs_GA", "SingleIs_RS"]:
    nb_best_trials = 40
if slurm_scenari in ["xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS"]:
    nb_best_trials = 1
    
output_path = "/beegfs/tferte/output/"
scenari_params_folder = output_path + slurm_scenari + "/*.csv"
csv_files = glob.glob(scenari_params_folder)
data_path="data_obfuscated/"

# evaluate algorithm depending on array
file_i = csv_files[int(array_id)]
subfolder = file_i.split('/')[-1].split('.')[0]
# get the date
min_date_eval = get_date_plus_14_from_subfolder(subfolder)
output_folder = output_path + slurm_scenari + "/test/" + min_date_eval + "/"
# Test algorithm
TestAlgorithm_for_csv(
  output_path = output_folder,
  data_path = data_path,
  study_path = file_i,
  nb_best_trials = nb_best_trials,
  nb_esn = nb_esn,
  lsTraining = [365],
  min_date_eval=min_date_eval
  )
