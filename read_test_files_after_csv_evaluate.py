import os
import pandas as pd
import glob

# Get the SLURM_ARRAY_TASK_ID from the environment
array_id = os.getenv("SLURM_ARRAY_TASK_ID")

# Define folder_path and folder_list
folder_path = "/beegfs/tferte/output/"
folder_list = ["GeneticSingleIs_GA_20esn", "GeneticSingleIs_GA_7", "GeneticSingleIs_GA_21", "prophet", "xgb_pred_RS_21", "xgb_pred_RS_7", "2000_units", "GeneticSingleIs_GA_1000", "GeneticSingleIs_GA", "GeneticSingleIs_GA_PCA" "GeneticSingleIs_RS", "SingleIs_GA", "enet_pred_RS", "xgb_pred_RS"]

# Get the folder_i based on array_id
folder_i = folder_list[int(array_id)]

# Create an empty list to store dataframes
df_list = []

# List all files in the specified folder
folder_path_glob = folder_path + folder_i + "/test/"
file_pattern = "**/*.csv"

files = glob.glob(os.path.join(folder_path_glob, file_pattern), recursive=True)

importance_files = [path for path in files if "_importance/" in path]
prediction_files = [path for path in files if "_importance/" not in path]

# Iterate over the files
for file in prediction_files:
  # Read the data from the file using pandas
  df_res = pd.read_csv(file)
  # Extract trial and hp_date from the file path
  string_x = file.split("/")
  trial = string_x[-2]
  hp_date = string_x[-3]
  # Add trial and hp_date as new columns
  df_res['trial'] = trial
  df_res['hp_date'] = hp_date
  # Append the dataframe to df_list
  df_list.append(df_res)

# Concatenate all dataframes in df_list into one
dfres = pd.concat(df_list, ignore_index=True)

# Save the resulting dataframe as an RDS file
# dfres.to_csv(folder_path + folder_i + "/" + folder_i + "_combined.csv", index = False)
dfres.to_csv("output/" + folder_i + "_combined.csv", index = False)

if folder_i not in ["GeneticSingleIs_GA_20esn", "GeneticSingleIs_GA_10esn"]:
    df_list = []
    # Iterate over the files
    for file in importance_files:
      # Read the data from the file using pandas
      df_res = pd.read_csv(file)
      # Extract trial and hp_date from the file path
      string_x = file.split("/")
      trial = string_x[-2]
      hp_date = string_x[-3]
      # Add trial and hp_date as new columns
      df_res['trial'] = trial.replace("_importance", "")
      df_res['hp_date'] = hp_date
      # Append the dataframe to df_list
      df_list.append(df_res)
    
    # Concatenate all dataframes in df_list into one
    dfres = pd.concat(df_list, ignore_index=True)
    
    # Save the resulting dataframe as an RDS file
    # dfres.to_csv(folder_path + folder_i + "/" + folder_i + "_combined.csv", index = False)
    dfres.to_csv("output/" + folder_i + "_importance_combined.csv", index = False)


