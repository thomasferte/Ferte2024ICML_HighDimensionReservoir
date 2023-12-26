import optuna
from train_test_api.utils import *

def TestAlgorithm(output_path = "/beegfs/tferte/simulations/test_set/small_test/",
  data_path = "/beegfs/tferte/data/",
  study_path = 'sqlite:////beegfs/tferte/database_storage/epidemio.db',
  study_name = '01_epidemio_common_is',
  nb_best_trials = 10,
  vecFeaturesEpi = ["hosp",
      "hosp_rolDeriv7",
      "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
      "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
      "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
      "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
      "IPTCC.mean",
      "Vaccin_1dose",
      "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"],
  nb_esn = 1,
  lsTraining = [30, 90, 180, 365, 1000],
  lsFiles = None
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
    study_name : string
        Study name
    nb_best_trials : integer
        Number of best trials used
    vecFeaturesEpi : list
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
    # get best trials
    study = optuna.create_study(study_name=study_name,
      storage = study_path,
      load_if_exists = True,
      direction='minimize')

    # Get all the trials as a DataFrame
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE']
    
    # Sort the trials based on the objective value in descending order
    sorted_trials = trials_df.sort_values('value', ascending=False)
    
    # Get the top 10 trials
    top_10_trials = sorted_trials.tail(nb_best_trials)
    trial_ids = top_10_trials['number'].tolist()
    # make a big list with anteriority
    full_list = [(mintraining, trial_id) for mintraining in lsTraining for trial_id in trial_ids]
    # evaluate trials
    for meta in full_list:
    	trial_id = meta[1]
    	mintraining = meta[0]
    	job_id = "trial_"+str(trial_id)+"_train"+str(mintraining)
    	# get trial
    	current_trial = study.trials[trial_id]
    	# hp
    	hp_keys_list = list(current_trial.params.keys())
    	
    	ridge = current_trial.params['ridge']
    	spectral_radius = current_trial.params['spectral_radius']
    	leaking_rate = current_trial.params['leaking_rate']
    	seed = None
    	bin_features = 0
    	alpha = 0
    	nb_features = 0
    	
    	if("seed" in hp_keys_list):
    		seed = current_trial.params['seed']
    	if("alpha" in hp_keys_list):
    		alpha = current_trial.params['alpha']
    	if("nb_features" in hp_keys_list):
    		nb_features = current_trial.params['nb_features']
    	
    	bin_list = [item for item in hp_keys_list if item.endswith('_bin')]
    	
    	if("input_scaling" in hp_keys_list):
    		input_scaling = current_trial.params['input_scaling']
    	else:
    		input_scaling = current_trial.params
    		entries_to_remove = ['ridge', 'spectral_radius', 'leaking_rate', 'alpha', 'nb_features'] + bin_list
    		for k in entries_to_remove:
    			temp = input_scaling.pop(k, None)
    	
    	# binary indicator for eah feature to be or not to be selected
    	if(len(bin_list) > 0):
    	  bin_features = {key: value for key, value in current_trial.params.items() if key in bin_list}
    	
    	fct_value = perform_full_training(
    		path=data_path,
    		lsFiles = lsFiles,
    		application_param=appParam(mintraining=mintraining,
    		    vecFeaturesEpi=vecFeaturesEpi,
    			  nb_esn= nb_esn,
    			  is_training=False),
    		reservoir_param=reservoirParam(
    			    units=500,
    			    ridge=ridge,
    			    seed=seed,
    			    input_scaling=input_scaling,
    			    leaking_rate=leaking_rate,
    			    bin_features=bin_features,
    			    spectral_radius=spectral_radius,
    			    alpha = alpha,
    			    nb_features = nb_features),
    		job_id=job_id,
    		output_path=output_path)
    
    return None
