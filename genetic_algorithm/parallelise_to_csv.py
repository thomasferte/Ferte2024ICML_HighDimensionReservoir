import fcntl
import json
import random
from train_test_api.utils import *
from genetic_algorithm.csv_class_genetic_algo import *
import numpy as np
import os
import pandas as pd
from datetime import datetime
from shutil import rmtree

##### functions

def define_hp_distribution(features, multiple_is = False, bin_features = False, enet = False, is_feature_selection = False, seed = False):
    # baseline dataframe    
    hp_df = ({
        'hp':["ridge","spectral_radius","leaking_rate"],
        'type_hp':["num", "num", "num"],
        'low' :[1e-10, 1e-5, 1e-5],
        'high':[1e5, 1e5, 1],
        'log':[True, True, True]
                  })
    hp_df = pd.DataFrame(hp_df)
    
    ## multiple input scaling
    if(multiple_is):
        for feature in features:
            hp_df.loc[len(hp_df)] = [feature, "num", 1e-5, 1e5, True]
    else :
        hp_df.loc[len(hp_df)] = ["input_scaling", "num", 1e-5, 1e5, True]
    
    ## binary feature selection
    if(bin_features):
        for feature in features:
            hp_df.loc[len(hp_df)] = [feature+"_bin", "binary", "n", "y", False]
    
    if(enet):
        hp_df.loc[len(hp_df)] = ["alpha", "num", 1e-4, 1e5, True]
    
    if(is_feature_selection):
        hp_df.loc[len(hp_df)] = ["nb_features", "int", 3, len(features), False]
    
    if(seed):
        hp_df.loc[len(hp_df)] = ["seed", "int", 1, 1e6, False]
    
    return hp_df

def random_sampler(type_hp, low, high, log):
      if(type_hp == "num"):
        if(log):
            res = 10**random.uniform(a=np.log10(low), b=np.log10(high))
        else :
            res = random.uniform(a=low, b=high)
      if(type_hp == "int"):
          res = random.randint(a=low, b=high)
      if(type_hp == "binary"):
          res = random.choice(seq=[low, high])
      
      return res

def random_sampler_from_hp_df(hp_df):
    res = {}
    for ind in hp_df.index:
        param_name = hp_df['hp'][ind]
        param_value = random_sampler(type_hp=hp_df['type_hp'][ind], low=hp_df['low'][ind], high=hp_df['high'][ind], log=hp_df['log'][ind])
        res[param_name] = param_value
    return res

def eval_objective_function(params, features, output_path, data_path, job_id, is_training=True, nb_esn = 3, rm_output_files = True, min_date_eval='2021-03-01', units = 500):
    # xgb
    if("n_estimators" in params.keys()):
        n_estimators = int(params["n_estimators"])
        model = "xgb"
        nb_esn = 1
    else :
        n_estimators = None
    if("max_depth" in params.keys()):
        max_depth = int(params["max_depth"])
    else :
        max_depth = None
    if("learning_rate" in params.keys()):
        learning_rate = float(params["learning_rate"])
    else :
        learning_rate = None
    if("subsample" in params.keys()):
        subsample = float(params["subsample"])
    else :
        subsample = None
    if("colsample_bytree" in params.keys()):
        colsample_bytree = float(params["colsample_bytree"])
    else :
        colsample_bytree = None
    # enet
    if("l1_ratio" in params.keys()):
        l1_ratio = float(params["l1_ratio"])
        model = "enet"
        nb_esn = 1
    else :
        l1_ratio = None
    if("alpha" in params.keys()):
        alpha = float(params["alpha"])
    else :
        alpha = 0
    # reservoir
    if("leaking_rate" in params.keys()):
        leaking_rate = float(params["leaking_rate"])
        model = "esn"
    else :
        leaking_rate = None
    if("spectral_radius" in params.keys()):
        spectral_radius = float(params["spectral_radius"])
    else :
        spectral_radius = None
    if("ridge" in params.keys()):
        ridge = float(params["ridge"])
    else :
        ridge = None
    if("seed" in params.keys()):
        seed = int(params["seed"])
        nb_esn = 1
    else :
        seed = None
    if("nb_features" in params.keys()):
        nb_features = int(params["nb_features"])
    else :
        nb_features = 0
    if("input_scaling" in params.keys()):
        input_scaling = float(params["input_scaling"])
    else :
        input_scaling = {key: float(value) for key, value in params.items() if key in features}
        if not bool(input_scaling) :
            input_scaling = None
    
    bin_filtered_dict = {key: value for key, value in params.items() if key.endswith("bin")}
    if(len(bin_filtered_dict) > 0):
        bin_features = bin_filtered_dict
    else :
        bin_features = 0

    fct_value = perform_full_training(
      path=data_path,
      min_date_eval=min_date_eval,
      application_param=appParam(mintraining=365, nb_esn= nb_esn, is_training=is_training,
                                vecFeaturesEpi=features),
      reservoir_param=reservoirParam(
        units=units,
        ridge=ridge,
        alpha=alpha,
        nb_features=nb_features,
        seed=seed,
        bin_features=bin_features,
        input_scaling=input_scaling,
        leaking_rate=leaking_rate,
        spectral_radius=spectral_radius,
        model = model,
        n_estimators = n_estimators,
        max_depth = max_depth,
        learning_rate = learning_rate,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        l1_ratio = l1_ratio),
      job_id=job_id,
      output_path=output_path)
    
    try:
        if rm_output_files:
            rmtree(output_path+job_id)
    except:
        pass
    
    # remove core files
    for filename in os.listdir():
        if filename.startswith("core"):
            os.remove(filename)
    
    return fct_value

def save_locked_csv(path_file, df_to_save):
    # if file didn't exist, read it to check and write with header if needed
    if(not os.path.exists(path_file)):
        with open(path_file, 'a+') as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
            # Check the number of lines
            file_size = file.tell()  # Move to the beginning of the file
            if file_size == 0:
                df_to_save.to_csv(path_file, index=False, mode = "a", header = True)
            else :
                df_to_save.to_csv(path_file, index=False, mode = "a", header = False)
            fcntl.flock(file, fcntl.LOCK_UN)
    # file did exist, write without header
    else :
        with open(path_file, "r+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0, 2)
            df_to_save.to_csv(path_file, index=False, mode = "a", header = False)
            fcntl.flock(file, fcntl.LOCK_UN)
    
    return None

def GA_or_randomsearch(path_file, Npop):
  res = []
  bool_file_exists = os.path.exists(path_file)
  if(bool_file_exists):
      with open(path_file, 'r') as file:
           fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
           perf_df = pd.read_csv(path_file, on_bad_lines = "skip")
           fcntl.flock(file, fcntl.LOCK_UN)
    
      perf_df = perf_df[perf_df['value'] != 'inprogress']
      perf_df = perf_df[perf_df['value'] != 'todo']
      if(len(perf_df) >= Npop):
          res = perf_df
  
  return res

def genetic_sampler_from_df(perf_df, hp_df, Npop, Ne):
    genetic_sampler = CsvGeneticAlgorithm(hp_df=hp_df, perf_df=perf_df, Npop = Npop, Ne = Ne)
    return genetic_sampler.sample_relative()

def scenari_define_hp_distribution(scenari, features):
    # test scenari available
    available_scenario = ['epidemio1Is', 'epidemioMultipleIs', 'Enet', 'GeneticSingleIs', 'GeneticMultipleIsBin', 'GeneticMultipleIsSelect', 'GeneticMultipleIsBinSeed', "xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS", "GeneticSingleIs_GA", "GeneticSingleIs_GA_1000", "GeneticSingleIs_RS", "SingleIs_GA", "SingleIs_RS", "GeneticSingleIs_GA_10esn"]
    if scenari not in available_scenario:
        raise ValueError("Scenari should be in " + ', '.join(available_scenario))
    
    multiple_is = scenari in ["epidemioMultipleIs", "GeneticMultipleIsBin", "GeneticMultipleIsSelect", "GeneticMultipleIsBinSeed"]
    bin_features = scenari in ["GeneticMultipleIsBin", "GeneticMultipleIsBinSeed", 'GeneticSingleIs', "GeneticSingleIs_GA", "GeneticSingleIs_GA_10esn", "GeneticSingleIs_GA_1000", "GeneticSingleIs_RS"]
    enet = scenari in ["Enet"]
    is_feature_selection = scenari in ["GeneticMultipleIsSelect"]
    seed = scenari in ["GeneticMultipleIsBinSeed"]
    
    if scenari in ["xgb_pred_GA", "xgb_pred_RS"] :
        hp_df = ({
        'hp':["n_estimators","max_depth","learning_rate", "subsample", "colsample_bytree"],
        'type_hp':["int", "int", "num", "num", "num"],
        'low' :[3, 5, 1e-5, 0, 0],
        'high':[300, 100, 1, 1, 1],
        'log':[False, False, True, False, False]
                  })
        hp_df = pd.DataFrame(hp_df)
    elif scenari in ["enet_pred_GA", "enet_pred_RS"] :
        hp_df = ({
        'hp':["ridge","l1_ratio"],
        'type_hp':["num", "num"],
        'low' :[1e-10, 0],
        'high':[1e5, 1],
        'log':[True, False]
                  })
        hp_df = pd.DataFrame(hp_df)
    else :
        hp_df = define_hp_distribution(
          features = features,
          multiple_is=multiple_is,
          bin_features=bin_features,
          enet = enet,
          is_feature_selection = is_feature_selection,
          seed = seed)
    
    return hp_df

def features_nbesn_optimizer_from_scenari(scenari):
    if scenari in ['Enet', 'GeneticSingleIs', 'GeneticMultipleIsBin', 'GeneticMultipleIsSelect', 'GeneticMultipleIsBinSeed',
    "xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS",
    "GeneticSingleIs_GA", "GeneticSingleIs_GA_10esn", "GeneticSingleIs_GA_1000", "GeneticSingleIs_RS",
    "SingleIs_GA", "SingleIs_RS"] :
        with open("data/allfeatures", "r") as fp:
            features = json.load(fp)
    else:
        features = ["hosp", "hosp_rolDeriv7",
                    "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                    "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                    "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                    "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                    "IPTCC.mean",
                    "Vaccin_1dose",
                    "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
    
    if scenari in ["xgb_pred_RS", "enet_pred_RS", "GeneticSingleIs_RS", "SingleIs_RS"] :
        global_optimizer = "RS"
    else :
        global_optimizer = "GA"
    
    ### set number of repetition
    if scenari in ["xgb_pred_RS", "enet_pred_RS", "GeneticSingleIs_RS", "SingleIs_RS"] :
        nb_esn = 1
    elif scenari in ["GeneticSingleIs_GA", "GeneticSingleIs_GA_1000", "GeneticSingleIs_RS", "SingleIs_GA", "SingleIs_RS"] :
        nb_esn = 3
    elif scenari in ["GeneticSingleIs_GA_10esn"] :
        nb_esn = 10
    
    return features, global_optimizer ,nb_esn

def csv_sampler(path_file, data_path, output_path, scenari, array_id = 1, Npop = 200, Ne = 100, nb_trials = 3200, date = '2021-03-01', units = 500):
    
    features, global_optimizer, nb_esn = features_nbesn_optimizer_from_scenari(scenari)
    
    ### Define hp distribution
    hp_df = scenari_define_hp_distribution(scenari, features)
    ### initiate nb_trials_done
    cpt = 0
    while cpt < 100:
        cpt +=1
        try:
            perf_df = GA_or_randomsearch(path_file=path_file, Npop=Npop)
        except:
            print("GA_or_randomsearch failed, retry")
            time.sleep(5)
    
    nb_trials_done = len(perf_df)
    ### launch loop
    cpt=-1
    while nb_trials_done < nb_trials:
        cpt += 1
        print("---- nb job evaluated = ", str(cpt) + "----")
        ### determine wether to use GA or random sampling
        perf_df = GA_or_randomsearch(path_file=path_file, Npop=Npop)
        nb_trials_done = len(perf_df)
        # choice of optimizer
        if global_optimizer == "GA" :
            if nb_trials_done > 0:
                optimizer = "GA"
                # keep only hp and value
                col_to_keep = hp_df.hp.to_list()
                col_to_keep.append("value")
                col_to_keep.append("job_id")
                filtered_perf_df = perf_df.loc[:, col_to_keep]
                params = genetic_sampler_from_df(perf_df=filtered_perf_df, hp_df=hp_df, Npop=Npop, Ne=Ne)
            else:
                optimizer = "RS"
                params = random_sampler_from_hp_df(hp_df=hp_df)
        else :
            optimizer = "RS"
            params = random_sampler_from_hp_df(hp_df=hp_df)
        ### evaluate objective function
        # define job_id
        job_start = datetime.now()
        job_id = "array_" + str(array_id) + "_trial_" + str(cpt) + "_time_" + job_start.strftime("%d_%m_%H_%M_%S")
        # evaluate
        value = eval_objective_function(params, features = features, data_path = data_path, job_id = job_id, output_path=output_path, min_date_eval=date, units = units, nb_esn=nb_esn)
        job_end = datetime.now()
        delta = job_end - job_start
        ### save results
        # prepare df to save
        params['value'] = value
        params['job_id'] = job_id
        params['time_seconds'] = delta.total_seconds()
        params['optimizer'] = optimizer
        df_to_save = pd.DataFrame.from_dict([params])
        ### save value + dictionnary params inside file
        save_locked_csv(path_file=path_file, df_to_save=df_to_save)
        nb_trials_done += 1
        
    return None

