from train_test_api.utils import *
from genetic_algorithm.parallelise_to_csv import *
from genetic_algorithm.monthly_update_from_csv import *
import os

##### define objective function #####
slurm_job = os.getenv('SLURM_ARRAY_JOB_ID')
# slurm_job = "TEST"
slurm_scenari = os.getenv('SLURM_JOB_NAME')
# slurm_scenari = "GeneticSingleIs_GA"
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
# array_id = 1

folder_path = "/beegfs/tferte/output/" + slurm_scenari + "/"
# folder_path = "output/" + slurm_scenari + "/"
data_path="data_obfuscated/"
# data_path="data_obfuscated_short/"
first_perf_file = slurm_scenari + "_" + str(slurm_job) + ".csv"
output_path = folder_path + "csv_parallel/"

units = 500
Npop = 200
Ne = 100
nb_trials_first = 3200

# Npop = 2
# Ne = 1
# nb_trials_first = 3

start_date = datetime(2021, 3, 1)
end_date = datetime(2022, 1, 1)
current_date = start_date

while current_date <= end_date:
    # Format the current date as 'YYYY-MM-DD'
    formatted_date = current_date.strftime("%Y-%m-%d")
    
    # Call your function with the formatted date
    print(formatted_date)
    csv_sampler(
      units = units,
      path_file= folder_path + formatted_date + ".csv",
      data_path=data_path,
      output_path= output_path+formatted_date+"/",
      scenari = slurm_scenari,
      array_id = str(array_id),
      Npop=Npop,
      Ne=Ne,
      nb_trials=nb_trials_first,
      date=formatted_date
      )
    
    # Move to the next month
    if current_date.month == 12:
        current_date = datetime(current_date.year + 1, 1, 1)
    else:
        current_date = datetime(current_date.year, current_date.month + 1, 1)



