import optuna
import pandas as pd
from train_test_api.utils import *
from genetic_algorithm.parallelise_to_csv import *
from re import sub
from re import match

def TestAlgorithm_for_csv(
  output_path,
  data_path,
  study_path,
  nb_best_trials = 40,
  units = 500,
  nb_esn = 1,
  lsTraining = [365],
  lsFiles = None,
  min_date_eval = '2021-03-01'
  ):
    """Evaluate best individuals from a study on the test set
    Parameters
    ----------
    output_path : string
        Output path
    data_path : string
        Data path
    study_path : string
        Path to study database
    nb_best_trials : integer
        Number of best trials used
    features : list
        List of features to be included in the model
    nb_esn : integer
        Number of reservoir to be trained by hp set
    lsTraining : list of integer
        Training anteriority to be used
    lsFiles : list
        list of files to evaluate
    Returns
    -------
    None
    """
    if "epidemio" in study_path :
        features = ["hosp", "hosp_rolDeriv7",
                    "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                    "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                    "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                    "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                    "IPTCC.mean",
                    "Vaccin_1dose",
                    "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
    else:
        with open("data/allfeatures", "r") as fp:
            features = json.load(fp)
    
    # Get all the trials as a DataFrame and remove trials with missing values
    trials_df = pd.read_csv(study_path).dropna()
    
    # Sort the trials based on the objective value in descending order and get top trials
    top_10_trials = trials_df.sort_values('value', ascending=False).tail(nb_best_trials)
    trial_ids = top_10_trials['job_id'].tolist()
    # make a big list with anteriority
    full_list = [(mintraining, trial_id) for mintraining in lsTraining for trial_id in trial_ids]
    # evaluate trials
    for meta in full_list:
    	trial_id = meta[1]
    	mintraining = meta[0]
    	job_id = "trial_"+str(trial_id)+"_train"+str(mintraining)
    	# get params
    	params = trials_df[trials_df["job_id"] == trial_id].to_dict(orient="records")[0]
    	temp = params.pop("value")
    	temp = params.pop("job_id")
    	# get trial
    	eval_objective_function(
    	  units = units,
    	  rm_output_files=False,
    	  params=params,
    	  features=features,
    	  output_path=output_path,
    	  data_path=data_path,
    	  job_id=job_id,
    	  nb_esn = nb_esn,
    	  is_training=False,
    	  min_date_eval=min_date_eval
    	  )
    
    return None


def get_date_plus_14_from_subfolder(subfolder):
  pattern = r'^\d{4}-\d{2}-\d{2}$'
  if match(pattern, subfolder):
    print(f"'{subfolder}' matches the 'YYYY-MM-DD' format.")
    temp_min_date_eval = subfolder
  else:
    print(f"'{subfolder}' does not match the 'YYYY-MM-DD' format.")
    temp_min_date_eval = '2021-03-01'
  # add 14 days to date to avoid overfit
  # Convert the min_date_eval to a datetime object
  date_obj = datetime.strptime(temp_min_date_eval, '%Y-%m-%d')
  # Add 14 days
  new_date_obj = date_obj + timedelta(days=14)
  # Format the result back to "YYYY-MM-DD" format
  min_date_eval = new_date_obj.strftime('%Y-%m-%d')
  
  print("Original Date:", temp_min_date_eval)
  print("Date after adding 14 days:", min_date_eval)
  
  return min_date_eval
