from genetic_algorithm.optuna_class_genetic_algo import *
from train_test_api.utils import *
import optuna
from genetic_algorithm.objective import objective_epidemio_common_is
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from test_algorithm.test_per_month import get_prediction_per_month



def create_new_study_from_old_result( Npop , Ne , study_path , date_to_eval , n_trials , 
                                     objective_function = objective_epidemio_common_is , nb_esn = 1 ,  top_trial = 2 , old_study_name = None):
    """Get list of files and order it
    Parameters
    ----------
    Npop : int
        Number of initial trial
    Ne : int
        Number of child to create
    study_path : string
        Path to the study stored
    date_to_eval : string
        String to determine the date until the training is performed
    n_trials : int
        Number of total trials for the genetic algrotihm
    objective_function : function
        Function to perform the optimisation
    old_study_name : 
        If None, get the first study name of the study stored
        If string, read specific study
    storage : string
        The path to database storage
    evolutive_param : evolutiveHpParam object
        An object with the evolution hp parameters
    Returns
    -------
    An optuna new study
    """
    study_summaries = optuna.study.get_all_study_summaries(storage= study_path)
    if old_study_name is False :
        study_name = study_summaries[0].study_name
    else:
        study_name = old_study_name
    study =  optuna.load_study(study_name= study_name, storage= study_path )
    # Get all the trials as a DataFrame
    trials_df = study.trials_dataframe()
    # Sort the trials based on the objective value in descending order
    sorted_trials = trials_df.sort_values('value', ascending=True)
    # Get the top 10 trials
    top_200_trials = sorted_trials.head(Npop)
    trial_ids = top_200_trials['number'].tolist()
    old_trials = study.get_trials(deepcopy=False)
    trial_to_copy = [old_trials[i] for i in trial_ids]
    new_trial = [
    optuna.trial.create_trial(
        params=trial.params,
        distributions=trial.distributions,
        state = optuna.trial.TrialState.WAITING
    ) for trial in trial_to_copy]
    sampler = CustomGeneticAlgorithm(Npop=Npop, Ne=Ne)
    new_study = optuna.create_study(study_name= date_to_eval,
                            storage = study_path,
                            load_if_exists = True,
                            sampler=sampler,
                            direction='minimize')


    new_study.add_trials(new_trial)
    min_date_eval = date_to_eval
    is_per_month = True
    new_study.optimize(lambda trial: objective_function(trial, min_date_eval), n_trials=n_trials)
    if (nb_esn <=3) & is_per_month:
        path = "./simulations/01_epidemio_common_is/output/" + min_date_eval +'/'
        top_trial_pred = get_prediction_per_month( path , min_date_eval , nb_esn , top_trial)
    return top_trial_pred


Npop = 200
Ne = 100
n_trials = 1200
study_path  = 'sqlite:///simulations/database_storage/test.db'

# date_to_eval = '2021-04-01'
# create_new_study_from_old_result( Npop , Ne , study_path, date_to_eval , n_trials = 6 , old_study_name = False)


# study_summaries = optuna.study.get_all_study_summaries(storage= study_path)
# study_name = study_summaries[0].study_name
# study =  optuna.load_study(study_name= study_name, storage= study_path )
# b = study.trials_dataframe()

# study =  optuna.load_study(study_name= '2021-04-01', storage= study_path )
# a = study.trials_dataframe()

# study =  optuna.load_study(study_name= '2021-05-01', storage= study_path )
# b = study.trials_dataframe()


first_date_to_eval = '2021-03-01'
last_date_to_eval = first_date_to_eval
date_to_eval = first_date_to_eval
while np.datetime64(date_to_eval) <= np.datetime64('2021-05-01'):
    if date_to_eval == first_date_to_eval:
        old_study_name = False
    else:
        old_study_name = last_date_to_eval
    date_to_eval = str(pd.to_datetime(last_date_to_eval)+pd.DateOffset(months=1))[:10]
    print(date_to_eval)
    print("****************NEW STUDY********")
    create_new_study_from_old_result( Npop , Ne , study_path, date_to_eval , n_trials = n_trials , old_study_name = old_study_name)
    last_date_to_eval = date_to_eval