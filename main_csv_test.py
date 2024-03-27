from train_test_api.utils import *
from test_algorithm.TestAlgorithm_for_csv import *
from genetic_algorithm.parallelise_to_csv import *
import os
import glob

slurm_scenari = os.getenv('SLURM_JOB_NAME')
# slurm_scenari = "GeneticSingleIs_GA"
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
# array_id = 10

print(slurm_scenari + " " + str(array_id))

# Get the params files
if slurm_scenari in ["GeneticSingleIs_GA", "GeneticSingleIs_GA_PCA", "GeneticSingleIs_RS", "SingleIs_GA", "SingleIs_RS", "GeneticSingleIs_GA_10esn", "GeneticSingleIs_GA_20esn", "GeneticSingleIs_GA_21", "GeneticSingleIs_GA_7"]:
    nb_best_trials = 40
    units = 500
if slurm_scenari in ["xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS", "xgb_pred_RS_21", "xgb_pred_RS_7", "prophet"]:
    nb_best_trials = 1
    units = 500
if slurm_scenari in ["GeneticSingleIs_GA_1000"]:
    nb_best_trials = 40
    units = 2000

output_path = "/beegfs/tferte/output/"
scenari_params_folder = output_path + slurm_scenari + "/*.csv"
csv_files = glob.glob(scenari_params_folder)

if slurm_scenari in ["GeneticSingleIs_GA_21", "xgb_pred_RS_21"]:
    data_path="data_obfuscated_forecast_21days/"
elif slurm_scenari in ["GeneticSingleIs_GA_7", "xgb_pred_RS_7"]:
    data_path="data_obfuscated_forecast_7days/"
else :
    data_path="data_obfuscated/"

# evaluate algorithm depending on array
file_i = csv_files[int(array_id)]
subfolder = file_i.split('/')[-1].split('.')[0]
# get the date
forecast_days, features, global_optimizer, nb_esn = features_nbesn_optimizer_from_scenari(slurm_scenari)
nb_esn = 1
min_date_eval = get_date_plus_14_from_subfolder(subfolder, forecast_days)
output_folder = output_path + slurm_scenari + "/test/" + min_date_eval + "/"
# Test algorithm
TestAlgorithm_for_csv(
  output_path = output_folder,
  data_path = data_path,
  study_path = file_i,
  nb_best_trials = nb_best_trials,
  nb_esn = nb_esn,
  lsTraining = [365],
  min_date_eval=min_date_eval,
  units = units
  )
