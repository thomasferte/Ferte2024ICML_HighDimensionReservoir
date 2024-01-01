import pandas as pd
from flock import Flock
from genetic_algorithm.parallelise_to_csv import *

def reevaluate_previous_trials(previous_perf_path, perf_folder, date, data_path, Npop, scenari, array_id, units = 500):
    print("get features")
    if scenari in ['Enet', 'GeneticSingleIs', 'GeneticMultipleIsBin', 'GeneticMultipleIsSelect', 'GeneticMultipleIsBinSeed',
    "xgb_pred_GA", "enet_pred_GA", "xgb_pred_RS", "enet_pred_RS",
    "GeneticSingleIs_GA", "GeneticSingleIs_RS"] :
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
    
    ##### select best trials from previous results
    print("get file perf = " + previous_perf_path)
    df_previous_perf = pd.read_csv(previous_perf_path).dropna()
    # with open(previous_perf_path, 'r') as file:
    #         fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
    #         df_previous_perf = pd.read_csv(file).dropna()
    #         fcntl.flock(file, fcntl.LOCK_UN)
    top_trials = df_previous_perf.sort_values('value', ascending=False).tail(Npop)
    top_trials["value"] = "todo"
    
    # if file didn't exist, read it to check and write with header if needed
    new_perf_file = perf_folder+date+".csv"
    # if file does not exist yet, create it with previous values to "todo"
    if(not os.path.exists(new_perf_file)):
            with open(new_perf_file, 'a+') as file:
                    fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
                    # Check the number of lines
                    file_size = file.tell()  # Move to the beginning of the file
                    if file_size == 0:
                        top_trials.to_csv(new_perf_file, index=False, mode = "a", header = True)
                    else :
                        top_trials.to_csv(new_perf_file, index=False, mode = "a", header = False)
                    fcntl.flock(file, fcntl.LOCK_UN)
            
#         with Flock(new_perf_file_lock, 'w'):
#             top_trials.to_csv(new_perf_file, index=False, mode = "w", header = True)
#         with open(new_perf_file, 'a+') as file:
#             fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
#             file_size = file.tell()  # Move to the beginning of the file
#             if file_size == 0:
#                 top_trials.to_csv(new_perf_file, index=False, mode = "w", header = True)
#             fcntl.flock(file, fcntl.LOCK_UN)
    
    # open file, set value to in progress, close file, evaluate job, update file
    nb_trials_to_reevaluate = 1
    while nb_trials_to_reevaluate>0:
        params = {}
        # try to write on file several times
        file_ok = 1
        while file_ok != 0 and file_ok < 100 :
            try:
                df_perf = pd.read_csv(new_perf_file)
                dftodo = df_perf[df_perf["value"] == "todo"]
                nb_trials_to_reevaluate = len(dftodo)
                print("nb_trials_to_reevaluate = " + str(nb_trials_to_reevaluate))
                if(nb_trials_to_reevaluate > 0):
                    random_row = dftodo.sample(n=1, random_state=random.seed())
                    job_id_to_do = random_row.iloc[0]["job_id"]
                file_ok = 0
                #with open(new_perf_file, 'a+') as file:
                #    fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
                #    df_perf = pd.read_csv(new_perf_file)
                #    dftodo = df_perf[df_perf["value"] == "todo"]
                #    nb_trials_to_reevaluate = len(dftodo)
                #    print("nb_trials_to_reevaluate = " + str(nb_trials_to_reevaluate))
                #    # set in progress value and save file
                #    if(nb_trials_to_reevaluate > 0):
                #        random_row = dftodo.sample(n=1, random_state=random.seed())
                #        job_id_to_do = random_row.iloc[0]["job_id"]
                #        # job_id_to_do = df_perf[df_perf["value"] == "todo"].iloc[0]["job_id"]
                #        # set to in progress to inform other nodes
                #        # df_perf.loc[df_perf["job_id"] == job_id_to_do, "value"] = "inprogress"
                #        # df_perf.to_csv(new_perf_file, index = False, mode = "w", header = True)
                #    # close file
                #    fcntl.flock(file, fcntl.LOCK_UN)
                #    file_ok = 0
            except:
                print(str(file_ok) + " failed attempt to access main file, retry")
                file_ok += 1
                time.sleep(5)
            
        # compiute the objective value for the job_id_todo
        if nb_trials_to_reevaluate > 0:
            value = 1000
            nb_try = 0
            while value > 999 and nb_try < 500:
              print("trial = " + str(nb_try))
              try:
                  params = df_perf[df_perf["job_id"] == job_id_to_do].to_dict(orient = "records")[0]
                  temp = params.pop("value")
                  temp = params.pop("job_id")
                  current_time = datetime.now().strftime("%d_%m_%H_%M_%S")
                  value = eval_objective_function(
                    units = units,
                    min_date_eval = date,
                    params = params,
                    features = features,
                    data_path = data_path,
                    job_id = job_id_to_do+"_at_" + date + "_by_" + array_id,
                    output_path=perf_folder+"csv_parallel/"+date+"/"
                  )
                  if value < 999:
                      with open(new_perf_file, 'a+') as file:
                          fcntl.flock(file, fcntl.LOCK_EX)  # Acquire an exclusive lock
                          df_perf = pd.read_csv(new_perf_file)
                          df_perf.dropna(inplace=True)
                          df_perf.loc[df_perf["job_id"] == job_id_to_do, "value"] = value
                          df_perf.loc[df_perf["job_id"] == job_id_to_do, "optimizer"] = "reevaluate"
                          df_perf.to_csv(new_perf_file, index = False, mode = "w", header = True)
                          fcntl.flock(file, fcntl.LOCK_UN)
              
              except pd.errors.EmptyDataError:
                  print("Failed to reevaluate objective function, retry")
                  value = 1000
                  nb_try += 1
                  time.sleep(2)
    
    return new_perf_file

def evolutive_hp_csv(array_id, perf_folder, first_perf_file, data_path, scenari, Npop = 200, Ne = 100, nb_trials = 1200, min_date_eval = datetime.strptime('2021-03-01', '%Y-%m-%d'), units = 500):
    ##### get all dates files
    files = pd.DataFrame(glob.glob(data_path + '*.csv'),columns = ['full_path'])
    files['file_name'] = files.full_path.str.split(data_path,n=1).str[-1]
    files['date'] = pd.to_datetime(files.file_name.str.split('.csv').str[0],format='%Y%m%d')
    files = files[files['date'] > min_date_eval]
    files['day'] = files['date'].dt.day
    files = files.sort_values("date")
    files = files.reset_index(drop=True)
    
    ##### iterate through date and reestimate hp if date day is 1 or 2
    previous_perf_path = perf_folder + first_perf_file
    for ind in files.index:
        day = files['day'][ind]
        date = files['date'][ind]
        date = date.strftime("%Y-%m-%d")
        if(day in [1,2]):
            print("------------------" + date + "---------------------")
            ### import previous results and reevaluate them
            trial_ok = 1
            while trial_ok != 0 and trial_ok < 1000 :
                try:
                    previous_perf_path = reevaluate_previous_trials(
                        units = units,
                        previous_perf_path=previous_perf_path,
                        perf_folder=perf_folder,
                        date=date,
                        data_path=data_path,
                        Npop=Npop,
                        array_id = array_id,
                        scenari=scenari
                        )
                    trial_ok = 0
                except pd.errors.EmptyDataError:
                    print(str(trial_ok) + " attempt, retry")
                    trial_ok += 1
                    time.sleep(1)
            ### GA for x interation with new min_date, isTraining = True and save results
            trial_sampler_ok = 1
            while trial_sampler_ok != 0 and trial_sampler_ok < 1000 :
                try:
                    csv_sampler(
                        units = units,
                        path_file=previous_perf_path,
                        date=date,
                        data_path=data_path,
                        output_path=perf_folder+"csv_parallel/"+date+"/",
                        scenari = scenari,
                        array_id = array_id,
                        Npop=Npop,
                        Ne=Ne,
                        nb_trials=nb_trials
                        )
                    trial_sampler_ok = 0
                except pd.errors.EmptyDataError:
                    print(str(trial_ok) + " attempt csv_sampler, retry")
                    trial_sampler_ok += 1
                    time.sleep(1)
            
            
    return None

# perf_folder = "output/"
# first_perf_file = "GeneticMultipleIsBin_11044201.csv"
# Npop = 2
# Ne = 1
# nb_trials = 3
# data_path = "data/"
# min_date_eval = datetime.strptime('2021-03-01', '%Y-%m-%d')
# scenari = "GeneticMultipleIsBin"
# array_id = 1
# evolutive_hp_csv(
#   array_id = array_id,
#   perf_folder = perf_folder,
#   first_perf_file = first_perf_file,
#   data_path = data_path,
#   scenari=scenari,
#   Npop = Npop,
#   Ne = Ne,
#   nb_trials = nb_trials,
#   min_date_eval = min_date_eval
# )
