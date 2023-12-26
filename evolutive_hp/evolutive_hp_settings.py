from train_test_api.utils import *
from evolutive_hp.evolutive_hp import *
import json

def evolutive_hp_settings(scenari, experience_name, output_path, data_path, storage, units = 500, Npop = 200, Ne = 100, nb_best_trials = 10, mintraining = 365, nb_esn = 10):
    # test scenari available
    available_scenario = ['epidemio1Is', 'epidemioMultipleIs', 'Enet', 'Genetic']
    if scenari not in available_scenario:
        raise ValueError("Scenari should be in " + ', '.join(available_scenario))
    
    # set evolutive_param
    evolutive_param = evolutiveHpParam(
        experience_name = experience_name,
        output_path = output_path,
        data_path = data_path,
        nb_best_trials = nb_best_trials,
        sampler = CustomGeneticAlgorithm(Npop=Npop, Ne=Ne),
        nbTrials = Npop+Ne*10,
        storage = storage
        )
    
    if scenari in ['epidemio1Is', 'epidemioMultipleIs']:
      application_param=appParam(
            mintraining=mintraining,
            nb_esn=nb_esn
            )
    
    elif scenari in ['Enet', 'Genetic']:
      with open("list_features/all457", "r") as fp:
              columns_to_keep = json.load(fp)

      application_param=appParam(
        mintraining=mintraining,
        nb_esn=nb_esn,
        vecFeaturesEpi=columns_to_keep
        )
    
    # define reservoir_param_fct and application_param
    if scenari == "epidemio1Is":
        def reservoir_param_fct(trial):
            res = reservoirParam(
                units=units,
                ridge=trial.suggest_float("ridge", 1e-10, 1e5, log=True),
                input_scaling=trial.suggest_float("input_scaling", 1e-5, 1e5, log=True),
                leaking_rate=trial.suggest_float("leaking_rate", 1e-3, 1, log=True),
                spectral_radius=trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True)
                )
            return res
        
    elif scenari == "epidemioMultipleIs":
        def reservoir_param_fct(trial):
            input_scaling = {}
            for column in application_param.vecFeaturesEpi:
                suggested_value = trial.suggest_float(column, 1e-5, 1e5, log=True)
                input_scaling[column] = suggested_value
            
            res = reservoirParam(
                units=units,
                ridge = trial.suggest_float("ridge", 1e-10, 1e5, log=True),
                spectral_radius = trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True),
                leaking_rate = trial.suggest_float("leaking_rate", 1e-5, 1, log=True),
                input_scaling = input_scaling
                )
            
            return res
      
    elif scenari == "Enet":
        def reservoir_param_fct(trial):
          res = reservoirParam(
              units=units,
              ridge = trial.suggest_float("ridge", 1e-10, 1e5, log=True),
              spectral_radius = trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True),
              leaking_rate = trial.suggest_float("leaking_rate", 1e-5, 1, log=True),
              input_scaling = trial.suggest_float("input_scaling", 1e-5, 1e5, log=True),
              alpha = trial.suggest_float("alpha", 1e-10, 1e5, log=True)
              )
          return res
      
    elif scenari == "Genetic":
      def reservoir_param_fct(trial):
        input_scaling = {}
        for column in columns_to_keep:
            suggested_value = trial.suggest_float(column, 1e-5, 1e5, log=True)
            input_scaling[column] = suggested_value
      
        res = reservoirParam(
            units=units,
            ridge = trial.suggest_float("ridge", 1e-10, 1e5, log=True),
            spectral_radius = trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True),
            leaking_rate = trial.suggest_float("leaking_rate", 1e-5, 1, log=True),
            input_scaling = input_scaling,
            nb_features = trial.suggest_int("nb_features", 3, len(columns_to_keep))
            )
        return res
    
    res_dict = {
      "evolutive_param" : evolutive_param,
      "application_param" : application_param,
      "reservoir_param_fct" : reservoir_param_fct
    }
    
    return res_dict
