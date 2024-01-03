from train_test_api.utils import *
from genetic_algorithm.parallelise_to_csv import *
from genetic_algorithm.monthly_update_from_csv import *
import os

##### define objective function #####
slurm_job = os.getenv('SLURM_ARRAY_JOB_ID')
# slurm_job = "test"
slurm_scenari = os.getenv('SLURM_JOB_NAME')
# slurm_scenari = "GeneticSingleIs_GA_1000"
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
# array_id = 1

folder_path = "/beegfs/tferte/output/" + slurm_scenari + "/"
# folder_path = "output/" + slurm_scenari + "/"
data_path="data_obfuscated/"
Npop = 200
Ne = 100
nb_trials_first = 3200
nb_trials_update = 1200
first_perf_file = slurm_scenari + "_" + str(slurm_job) + ".csv"
output_path = folder_path + "csv_parallel/"

if slurm_scenari in ["GeneticSingleIs_GA_1000"]:
    units = 1000
else :
    units = 500

# Npop = 2
# Ne = 1
# nb_trials_first = 3
# nb_trials_update = 3

print("------- first optimisation ------------")
csv_sampler(
  units = units,
  path_file= folder_path + first_perf_file,
  data_path=data_path,
  output_path= output_path+"first_optimisation/",
  scenari = slurm_scenari,
  array_id = str(array_id),
  Npop=Npop,
  Ne=Ne,
  nb_trials=nb_trials_first
  )

print("------- monthly update ------------")
evolutive_hp_csv(
  units = units,
  array_id = str(array_id),
  perf_folder = folder_path,
  first_perf_file = first_perf_file,
  data_path = data_path,
  scenari=slurm_scenari,
  Npop = Npop,
  Ne = Ne,
  nb_trials = nb_trials_update
)
