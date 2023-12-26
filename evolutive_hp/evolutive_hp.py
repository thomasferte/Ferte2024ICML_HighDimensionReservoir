import optuna
import pandas as pd
import glob
import os
from train_test_api.utils import *
from genetic_algorithm.optuna_class_genetic_algo import *
from test_algorithm.TestAlgorithm import *

def evolutiveHp(evolutive_param, reservoir_param_fct, application_param):
    """Evaluate best individuals from a study on the test set
    Parameters
    ----------
    data_path : string
        Data path
    Returns
    -------
    None
    """
    ### optimise algorithm on first day, return current_study
    # get date and files
    dict_row = getInfoFromRow(evolutive_param.files.iloc[0], evolutive_param)
    # define objective function
    objective = generateObjectiveFunction(evolutive_param=evolutive_param,
          lsFiles = [dict_row["file_shift_i"]],
          application_param = application_param,
          reservoir_param_fct = reservoir_param_fct,
          output_path = dict_row["output_path_optimise_i"])
    # create study
    print("optimise performance up to " + dict_row["date"])
    current_study = createStudyOptimise(previous_study = None,
                        new_study_name = dict_row["path_name_i"],
                        evolutive_param = evolutive_param,
                        objective = objective)
    # remove core files                    
    file_paths = glob.glob("core.*")
    for file_path in file_paths:
        os.remove(file_path)
    
    ls_files_optimise = []
    ls_files_evaluate = []
    
    for index, row in evolutive_param.files.iterrows() :
        # get info from files
        dict_row = getInfoFromRow(row, evolutive_param)
        # intialise list of files for hp optimisation and evaluation
        ls_files_optimise.append(dict_row["file_shift_i"])
        ls_files_evaluate.append(dict_row["file_i"])
        # stop to evaluate and optimise at first days of month
        if(dict_row["day"] in [1, 2] or index == evolutive_param.files.index.max()):
            print("eval performance up to " + dict_row["date"])
            ### eval performance on ls_files_evaluate
            TestAlgorithm(output_path = dict_row["output_path_eval_i"],
                          data_path = evolutive_param.data_path,
                          study_path = evolutive_param.storage,
                          study_name = current_study.study_name,
                          nb_best_trials = evolutive_param.nb_best_trials,
                          vecFeaturesEpi = application_param.vecFeaturesEpi,
                          nb_esn = application_param.nb_esn,
                          lsTraining = [application_param.mintraining],
                          lsFiles = ls_files_evaluate)
            # reset ls_files_evaluate
            ls_files_evaluate = []
            ### optimise hyperparameters on ls_files_optimise
            if(index != evolutive_param.files.index.max()):
                print("optimise performance up to " + dict_row["date"])
                # define objective function
                objective = generateObjectiveFunction(evolutive_param=evolutive_param,
                      lsFiles = ls_files_optimise,
                      application_param = application_param,
                      reservoir_param_fct = reservoir_param_fct,
                      output_path = dict_row["output_path_optimise_i"])
                # create study
                current_study = createStudyOptimise(previous_study = current_study,
                                    new_study_name = dict_row["path_name_i"],
                                    objective = objective,
                                    evolutive_param=evolutive_param)
            
            # remove core files
            file_paths = glob.glob("core.*")
            for file_path in file_paths:
                os.remove(file_path)
    
    return None

def getfiles(data_path, shift = 7):
    """Get list of files and order it
    Parameters
    ----------
    data_path : string
        Data path
    Returns
    -------
    files, a pandas dataframe with date, day and date shifted column for
    optimisation of hyperparameters
    """
    files = pd.DataFrame(glob.glob(data_path + '*.csv'),columns = ['full_path'])
    files['file_name'] = files.full_path.str.split(data_path,n=1).str[-1]
    files['date'] = pd.to_datetime(files.file_name.str.split('.csv').str[0],format='%Y%m%d')
    files['day'] = files['date'].dt.day
    files = files.sort_values(by='date')
    files['full_path_shift14'] = files['full_path'].shift(shift)
    files = files.dropna().reset_index()
    
    return files

def createStudyOptimise(new_study_name, objective, evolutive_param, previous_study = None):
    """Get list of files and order it
    Parameters
    ----------
    data_path : string
        Data path
    previous_study : optuna study object
        Previous study from which to enqueue trials default is None
    new_study_name : string
        The new study name
    storage : string
        The path to database storage
    evolutive_param : evolutiveHpParam object
        An object with the evolution hp parameters
    Returns
    -------
    An optuna new study
    """
    # create study
    new_study = optuna.create_study(study_name=new_study_name,
                                storage = evolutive_param.storage,
                                load_if_exists = True,
                                sampler=evolutive_param.sampler,
                                direction='minimize')
    ### append previous trials
    if(previous_study != None):
        # Get all the trials as a DataFrame
        trials_df = previous_study.trials_dataframe()
        trials_df = trials_df[trials_df['state'] == 'COMPLETE']
        # Sort the trials based on the objective value in descending order
        sorted_trials = trials_df.sort_values('value', ascending=False)
        # Get the top Npop trials
        top_trials = sorted_trials.tail(evolutive_param.sampler.Npop)
        trial_ids = top_trials['number'].tolist()
        # enqueue trials
        for trial_i in trial_ids :
            new_study.enqueue_trial(previous_study.trials[trial_i].params)
        # optimise
    new_study.optimize(objective, n_trials=evolutive_param.nbTrials)
    
    return new_study
    
def generateObjectiveFunction(evolutive_param, lsFiles, application_param, reservoir_param_fct, output_path):
    
    def objective(trial):
        fct_value = perform_full_training(
            path=evolutive_param.data_path,
            lsFiles = lsFiles,
            application_param=application_param,
            reservoir_param=reservoir_param_fct(trial),
            job_id=str(trial.number),
            output_path=output_path)
        return fct_value
      
    return objective

def getInfoFromRow(row, evolutive_param):
    # get info from files
    day = row["day"]
    date = row["date"].strftime('%Y%m%d')
    file_i = row["full_path"]
    file_shift_i = row["full_path_shift14"]
    path_name_i = evolutive_param.experience_name+"_"+date
    output_path_eval_i = evolutive_param.output_path + "evaluate/" + path_name_i + "/"
    output_path_optimise_i = evolutive_param.output_path + "optimise/" + path_name_i + "/"
    
    my_dict = {
    "day": day,
    "date": date,
    "file_i": file_i,
    "file_shift_i": file_shift_i,
    "path_name_i": path_name_i,
    "output_path_eval_i": output_path_eval_i,
    "output_path_optimise_i": output_path_optimise_i
    }
    
    return my_dict

class evolutiveHpParam(object):
    def __init__(self,
                experience_name,
                output_path,
                nb_best_trials,
                sampler = CustomGeneticAlgorithm(Npop=200, Ne=100),
                nbTrials = 200+100*10,
                storage = 'sqlite:///sandbox/test.db',
                data_path = "data/",
                shift = 7
                ):
        self.data_path = data_path
        self.experience_name = experience_name
        self.output_path = output_path
        self.nb_best_trials = nb_best_trials
        self.sampler = sampler
        self.nbTrials = nbTrials
        self.storage = storage
        self.shift = shift
        # compute
        self.files = getfiles(self.data_path, shift=self.shift)

