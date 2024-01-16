import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime,timedelta
import reservoirpy as reservoirpy
from multiprocessing import Process, current_process
import multiprocessing
import psutil
from itertools import repeat
import time
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.decomposition import PCA


# Reservoir parameters
class reservoirParam(object):
    """An object for defining all the parameters used in reservoir computing.
.. rubric:: Attributes
units : (int,default to 500)
    Number of reservoir units
warmup : (int, default to 30)
     Number of timesteps to consider as warmup and discard at the begining of each timeseries before training.
input_connectivity : (float, default to 0.1)
    Connectivity of input neurons, i.e. ratio of input neurons connected to reservoir neurons. Must be in ]0,1]
rc_connectivity : (float, default to 0.1)
    Connectivity of recurrent weight matrix, i.e. ratio of reservoir neurons connected to other reservoir neurons, including themselves. Must be in ]0,1].
ridge : (float, default to 1e-2) 
    L2 regularization parameter.
spectral_radius : (float)
    Spectral radius of recurrent weight matrix.    
input_scaling :  (float or array-like of shape (features,), default to 2000.)
    Input gain. An array of the same dimension as the inputs can be used to set up different input scaling for each feature.
leaking_rate :  (float, default to 1.0) 
    Neurons leak rate. Must be in ]0,1].
activation :  (str or callable, default to tanh())
    Reservoir units activation function. - If a str, should be a activationsfunc function name. - If a callable, should be an element-wise operator on arrays.
seed : (int or numpy.random.Generator, optional)
    A random state seed, for noise generation.
inputBias : (bool, default to True)
    If False, no bias is added to inputs.
alpha : (float, default to 0)
    Elasticnet penalisation for feature selection.
nb_features : (int, default to 0)
    Number of features to be selected.
"""
    def __init__(self, nb_features = 0, alpha = 0, units =500, warmup = 30 , input_connectivity = 0.1 , rc_connectivity = 0.1 , ridge = 1e-2, spectral_radius = 50 ,
                 input_scaling = 2000, leaking_rate = 1 , bin_features = 0, activation = "tanh", seed = None ,inputBias = True,
                 model = "esn",
                 n_estimators = 10, max_depth = 10, learning_rate = 0.1, subsample = 0.3, colsample_bytree = 0.3,
                 l1_ratio = 0.5,
                 pca = 0):
        # define model
        self.model = model
        # reservoir hp
        self.units = units
        self.warmup = warmup    
        self.input_connectivity=  input_connectivity
        self.rc_connectivity =  rc_connectivity
        self.ridge = ridge
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.activation = activation
        self.seed = seed
        self.inputBias = inputBias
        self.nb_features = nb_features
        self.bin_features = bin_features
        # xgb hp
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        # enet hp
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        # pca
        self.pca = pca
        
        # chose between selection by enet or genetic
        if nb_features != 0 and alpha != 0: 
            raise Exception("nb_features (genetic algo selection) and alpha (elastic net selection) are different from 0, chose between them")
        
        
# Application parameters
class appParam(object):
    def __init__(self, vecFeaturesEpi = ["hosp", "hosp_rolDeriv7",
                   "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                   "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                   "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                   "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                   "IPTCC.mean",
                   "Vaccin_1dose",
                   "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"],
                mintraining = 335 , warmup = 30 , nb_esn = 10, doEnet = False , beta_linear_enet_is = 0, beta_sqrt_enet_is = 0, name = None, is_training = True):
        self.mintraining = mintraining
        self.warmup = warmup
        self.minanteriorite = self.warmup + self.mintraining
        self.nb_esn = nb_esn
        self.vecFeaturesEpi = vecFeaturesEpi
        self.doEnet = doEnet
        self.beta_linear_enet_is = beta_linear_enet_is
        self.beta_sqrt_enet_is = beta_sqrt_enet_is
        self.name = name
        self.is_training = is_training
        
        # check vecfeature type
        if not isinstance(vecFeaturesEpi, list): 
            raise Exception("vecFeaturesEpi must be a list (e.g ['all']")
        if len(vecFeaturesEpi) == 1 and vecFeaturesEpi[0] != "all": 
            raise Warning("If vecFeaturesEpi is of length 1, it must be equal to ['all'] to select all features")
        
def select_features_from_names(df, reservoir_param, application_param):
    """Select features on dataframe based on list of names
    Parameters
    ----------
    df : dataframe
        Raw data
    reservoir_param : reservoirParam object
        Object generated by reservoirParam
    application_param : appParam
        Object to define the global application parameters
    Returns
    -------
    df_selected : dataframe
        dataframe with selected features and outcome & outcomeDate
    """
    vecFeatures = application_param.vecFeaturesEpi
    column_names = df.columns.tolist()
    # Retrieve the vecFeatures list if 'all' provided
    if ((len(vecFeatures) == 1 ) & (vecFeatures[0] == "all")):
        vecFeatures = [col for col in column_names if col not in ['outcome','outcomeDate']]
    
    # get intersection between features in list and in dataframe
    vecFeatures = [c for c in vecFeatures if c in column_names]
    
    # decrease reservoir_param.nb_features if its greater than vecFeatures length
    if(reservoir_param.nb_features != 0 and len(vecFeatures) < reservoir_param.nb_features) :
        reservoir_param.nb_features = len(vecFeatures)
    
    # get the top features if nb_features != 0
    if(reservoir_param.nb_features != 0):
        # Sort vecFeatures based on the scores in features dictionary
        sorted_vecFeatures = sorted(vecFeatures, key=lambda x: reservoir_param.input_scaling[x], reverse=True)
        # Select the top 2 features
        vecFeatures = sorted_vecFeatures[:reservoir_param.nb_features]
    
    # select the features based on binary hyperparameters
    if(reservoir_param.bin_features != 0):
        # get the features with bin_features == 1
        keys_with_value_one = []
        # Loop through the dictionary items and get features where value == 1
        for key, value in reservoir_param.bin_features.items():
            key_cleaned = key.replace('_bin', '')
            if value == "y" and key_cleaned in vecFeatures:
                keys_with_value_one.append(key_cleaned)
        # Select all the features 
        vecFeatures = keys_with_value_one
    
    # select the columns from dataframe
    df_selected= df[vecFeatures + ['outcome','outcomeDate']]
    # remove constant features
    df_selected = df_selected.loc[:,df_selected.apply(pd.Series.nunique) != 1]
    
    # elastic net feature selection if needed
    if(reservoir_param.alpha != 0):
        # prepare data
        dfenet = df_selected.dropna()
        X = dfenet.drop(["outcome", "outcomeDate"],axis=1)
        Y = dfenet["outcome"]
        # scale X data
        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(X)
        # Create a Lasso regression model
        elastic_net = ElasticNet(alpha=reservoir_param.alpha, l1_ratio=0.5)
        elastic_net.fit(X=Xscaled, y=Y)
        # Get the selected features based on non-zero coefficients
        FeaturesPosition = [feature for feature, coef in zip(range(X.shape[1]), elastic_net.coef_) if coef != 0]
        vecFeatures = [X.columns[i] for i in FeaturesPosition]
        # select the columns from dataframe
        df_selected= df_selected[vecFeatures + ['outcome','outcomeDate']]
    
    return df_selected

def standardise_data_for_ens(df, norm_array = None):
    """Standardise the data in order to perform reservoir training
    Parameters
    ----------
    df : dataframe
        Data to be standardise
    norm_array : array
        If exist, use this array to perform the scaling, otherwise compute it
    Returns
    -------
    X_norm : array
        Input array standardise for reservoir training
    Y_norm : array
        Output array for reservoir training
    scaling : dict
        a dictonnary with the features use before normalisation, and the scaling array computed

    """
    Y = np.zeros((df.shape[0],1))
    Y[:,0] = np.array(df.outcome)
    X = np.array(df.drop(['outcome','outcomeDate'],axis = 1))
    if norm_array is None:
        norm_array = np.max(np.abs(X),axis = 0)
    X_norm = X/norm_array
    scaling = {"features": df.drop(['outcome','outcomeDate'],axis = 1), "vecMaxAbs": norm_array}
    return X_norm, Y, scaling

def fit_enet(X, Y, reservoir_param):
    enet_model = ElasticNet(alpha=reservoir_param.ridge, l1_ratio=reservoir_param.l1_ratio)
    enet_model.fit(X,Y)
    return enet_model

def fit_xgboost(X, Y, reservoir_param):
    xgb_model = xgb.XGBRegressor(n_estimators = reservoir_param.n_estimators,
      max_depth = reservoir_param.max_depth,
      learning_rate = reservoir_param.learning_rate,
      subsample = reservoir_param.subsample,
      colsample_bytree = reservoir_param.colsample_bytree,
      n_jobs = -1)
    xgb_model.fit(X,Y)
    return xgb_model

def fit_esn(X,Y, reservoir_param, application_param, vec_coef_enet = 0):
    """Select features on dataframe based on list of names
    Parameters
    ----------
    X : array
        Input array for reservoir training
    Y : array
        Output array for reservoir training
    reservoir_param : reservoirParam
        Object to definie parameters of the reservoir
    application_param : appParam
        Object to define the global application parameters
    vec_coef_enet : (float or array-like of shape (features,), default to 0.)
        If not egal to zero, define the input scaling based on the elastic net decomposition
    Returns
    -------
    esn : reservoirpy node
        Return the ESN trained on the (X,Y) 
    """
    reservoirpy.verbosity(0)
    if application_param.doEnet:
        input_scaling = reservoir_param.input_scaling + application_param.beta_linear_enet_is*np.abs(np.array(vec_coef_enet))+application_param.beta_sqrt_enet_is*np.sqrt(np.abs(np.array(vec_coef_enet)))
    else:
        input_scaling = reservoir_param.input_scaling
    if reservoir_param.seed is None:
        node = reservoirpy.nodes.Reservoir(units = reservoir_param.units,
                                            lr = reservoir_param.leaking_rate,
                                            sr = reservoir_param.spectral_radius,
                                            name = application_param.name,
                                            input_bias = reservoir_param.inputBias,
                                            input_scaling = reservoir_param.input_scaling,
                                            rc_connectivity = reservoir_param.rc_connectivity,
                                            input_connectivity = reservoir_param.input_connectivity,
                                            activation = reservoir_param.activation
                                            )
    else:
        node = reservoirpy.nodes.Reservoir(units = reservoir_param.units,
                                            lr = reservoir_param.leaking_rate,
                                            sr = reservoir_param.spectral_radius,
                                            name = application_param.name,
                                            input_bias = reservoir_param.inputBias,
                                            input_scaling = reservoir_param.input_scaling,
                                            rc_connectivity = reservoir_param.rc_connectivity,
                                            input_connectivity = reservoir_param.input_connectivity,
                                            activation = reservoir_param.activation,
                                            seed = reservoir_param.seed
                                            )
    ridge_layer = reservoirpy.nodes.Ridge(name = application_param.name,
                                         ridge = reservoir_param.ridge,
                                         input_bias = reservoir_param.inputBias)
    source = reservoirpy.nodes.Input()
    esn = [source >> node, source] >> ridge_layer
    esn.fit(X,Y, warmup = reservoir_param.warmup)
    return esn

def pref_on_test_set(dftest, selected_columns, reservoir_param , application_param , norm_array, esn, pca_model = None):
    dftest_selected = dftest[selected_columns]
    X_esn, Y_esn, scaling = standardise_data_for_ens(dftest_selected, norm_array )
    if pca_model != None :
        X_esn = pca_model.transform(X_esn)
    if reservoir_param.model == "esn" :
        vecPred = esn.run(X_esn , reset = True)
    if reservoir_param.model in ["enet","xgb"] :
        vecPred = esn.predict(X_esn)
    dfres = dftest.copy()
    dfres['pred'] = np.squeeze(vecPred) + dfres['hosp']
    dfres = dfres[['outcomeDate','outcome','hosp','pred']].tail(1)
    dfres['nbFeatures'] = np.shape(X_esn)[1]
    dfres['model'] = reservoir_param.model
    dfres['mintraining'] = application_param.mintraining
    return dfres

    
def task(index, selected_files,application_param,reservoir_param,output_path,job_id):
    selected_files = selected_files.reset_index()
    print(str(index) + ' is done by ' + current_process().name + ' on cpu ' + str(psutil.Process().cpu_num()))
    file_i = selected_files.full_path[index]
    input_data = pd.read_csv(file_i)
    dftrain = input_data.copy()
        
    dftrain = dftrain[dftrain['outcomeDate'] <= max(dftrain['START_DATE'])]
    dftrain.outcome= dftrain.loc[:,"outcome"].values-dftrain.loc[:,"hosp"].values
    # Remove some columns
    dftrain = dftrain.drop(['START_DATE'],axis=1)
    dftest = input_data
    num_lines = len(dftrain)  # Get the number of lines in dftrain
    if (application_param.minanteriorite!=0 and application_param.minanteriorite < num_lines):
        # Select anteriority
        dftrain = dftrain.tail( n = application_param.minanteriorite )
        dftest = dftest.tail( n = application_param.minanteriorite )
    # Select features use in training
    df_select = select_features_from_names(df=dftrain,reservoir_param=reservoir_param,application_param=application_param)
    selected_columns = df_select.columns
    # Do normalisation for ESN
    X_esn, Y_esn, scaling = standardise_data_for_ens(df=df_select)
    norm_array = scaling['vecMaxAbs']
    # Update input scaling on remaining features in the correct order
    if isinstance(reservoir_param.input_scaling, dict):
        remainingcols = scaling['features'].columns
        input_scaling_list = []
        # for each column in remaining columns, get the corresponding inputscaling and append to the list
        for column in remainingcols:
            input_scaling_list.append(reservoir_param.input_scaling[column])
        reservoir_param.input_scaling = np.array(input_scaling_list)

    pred_esn = pd.DataFrame(columns = ["outcomeDate", "outcome", "hosp","pred","nbFeatures","model", "mintraining"])
    importance = None
    current_outcomeDate = dftest['outcomeDate'].tail(1).to_list()[0]
    
    # PCA if needed
    if reservoir_param.pca != 0 :
            pca = PCA(reservoir_param.pca)
            pca_model = pca.fit(X_esn)
            X_esn = pca_model.transform(X_esn)
    else :
        pca_model = None
    
    for j in range(application_param.nb_esn):
        ### Train the ESN
        if reservoir_param.model == "esn" :
            trained_esn = fit_esn(X_esn,Y_esn, reservoir_param, application_param)
            if not application_param.is_training:
                ridge_layer = trained_esn.node_names[3]
                coef = trained_esn.get_param(ridge_layer)['Wout'].tolist()
                coef = [item for sublist in coef for item in sublist]
                nb_param = len(coef)+len(trained_esn.get_param(ridge_layer)['bias'])
                
                concat_layer = trained_esn.node_names[2]
                concat_node = trained_esn.get_node(concat_layer)
                reservoir_features = ["reservoir" + str(i) for i in range(0, reservoir_param.units)]
                if reservoir_param.pca != 0:
                    input_features = ["pca" + str(i) for i in range(0, np.shape(X_esn)[1])]
                else :
                    input_features = scaling['features'].columns.to_list()
                if concat_node.input_dim[0] == reservoir_param.units :
                    features_list = reservoir_features + input_features
                elif concat_node.input_dim[1] == reservoir_param.units :
                    features_list = input_features + reservoir_features
        
        if reservoir_param.model == "enet" :
            trained_esn = fit_enet(X_esn, Y_esn, reservoir_param)
            if not application_param.is_training:
                features_list = scaling['features'].columns.to_list()
                coef = trained_esn.coef_.tolist()
                nb_param = len(coef)+len(trained_esn.intercept_)
                    
        if reservoir_param.model == "xgb" :
            trained_esn = fit_xgboost(X_esn, Y_esn, reservoir_param)
            if not application_param.is_training:
                # get feature importance of xgboost
                features_list = scaling['features'].columns.to_list()
                coef = trained_esn.feature_importances_.tolist()
                nb_param = trained_esn._Booster.trees_to_dataframe().shape[0]
        
        if not application_param.is_training:
            importance_i = pd.DataFrame({'iter': j, 'features': features_list, 'importance': coef, 'nb_param': nb_param, 'outcomeDate': current_outcomeDate})
            importance = pd.concat([importance, importance_i], ignore_index=True)
                
        # Predict on new data
        pred_j_esn = pref_on_test_set(dftest=dftest,
          selected_columns = selected_columns,
          reservoir_param=reservoir_param ,
          application_param=application_param ,
          norm_array=norm_array,
          esn=trained_esn,
          pca_model = pca_model)
        pred_esn = pd.concat([pred_esn, pred_j_esn], ignore_index=True)
    
    if not application_param.is_training:
        os.makedirs(output_path + job_id + '_importance', exist_ok = True)
        importance.to_csv(output_path+ job_id + '_importance/result_job_'+ selected_files.file_name[index],index=False)

    pred_esn.to_csv(output_path+ job_id +'/result_job_'+ selected_files.file_name[index],index=False)
    

def get_file_to_do(files, files_done):
    
    if files_done.shape[0]>0:
        files_to_do = files.copy()
        for i in range(files_done.shape[0]):
            #files_to_do = files_to_do.drop(files_to_do[files.file_name == files_done.file_name[i]].index)
            
            files_to_do.drop(files_to_do[files_to_do['file_name'] == files_done.file_name[i]].index, inplace = True)
        return files_to_do
            
    else:
        return files

def get_relative_baseline(path_out):
    files_result = glob.glob(path_out + '/*.csv')
    
    if len(files_result) == 0 :
        return 1000
    
    temp = []

    for filename in files_result:
        df = pd.read_csv(filename, index_col=None, header=0)
        temp.append(df)

    df_result = pd.concat(temp, axis=0, ignore_index=True)
    df_result.loc[df_result['pred']<0,'pred'] = 0
    df_grouped_by_date = df_result.groupby(['outcomeDate', 'model']).median()[['outcome','hosp','pred']]
    df_filter = df_grouped_by_date.loc[~(df_grouped_by_date['outcome'].isna())].copy()
    # replace by 10 if below 10 (obfuscation)
    df_filter['pred'] = df_filter['pred'] * (df_filter['pred'] > 10) + 10 * (df_filter['pred'] <= 10)
    df_filter['outcome'] = df_filter['outcome'] * (df_filter['outcome'] > 10) + 10 * (df_filter['outcome'] <= 10)
    df_filter['hosp'] = df_filter['hosp'] * (df_filter['hosp'] > 10) + 10 * (df_filter['hosp'] <= 10)
    # compute error
    df_filter['AE'] = np.abs(df_filter['pred']-df_filter['outcome'])
    df_filter['AE_baseline'] = np.abs(df_filter['hosp']-df_filter['outcome'])
    df_filter['delta_baseline'] = df_filter['AE']-df_filter['AE_baseline']
    df_filter['relative_baseline']=df_filter['AE']/df_filter['AE_baseline']
    df_filter['RE'] = df_filter['AE']/df_filter['outcome']

    return df_filter['AE'].mean()


def perform_full_training(path, application_param, reservoir_param, job_id,output_path, min_date_eval='2021-03-01', forecast_days=14, lsFiles = None):
    if reservoir_param.nb_features != 0 and (not isinstance(reservoir_param.input_scaling, dict)): 
        raise Exception("nb_features is different from 0, input_scaling is expected to be a dictionary")
    
    files = pd.DataFrame(glob.glob(path + '*.csv'),columns = ['full_path'])
    files['file_name'] = files.full_path.str.split(path,n=1).str[-1]
    # Get Date of the file
    files['date'] = pd.to_datetime(files.file_name.str.split('.csv').str[0],format='%Y%m%d')
    files = files.sort_values(by='date').reset_index()
    # min_date_eval = '2021-03-01'
    min_date_eval = datetime.strptime(min_date_eval, '%Y-%m-%d') - timedelta(days=forecast_days)
    # Selection by date
    if application_param.is_training:
        selected_files= files[files['date']<min_date_eval]
    else:
        selected_files= files[files['date']>=min_date_eval]
    if lsFiles != None:
        selected_files = pd.DataFrame(lsFiles,columns = ['full_path'])
        selected_files['file_name'] = selected_files.full_path.str.split(path,n=1).str[-1]
        # Get Date of the file
        selected_files['date'] = pd.to_datetime(selected_files.file_name.str.split('.csv').str[0],format='%Y%m%d')
        selected_files = selected_files.sort_values(by='date').reset_index()
    
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(output_path + job_id, exist_ok = True)
    
    batch_size = multiprocessing.cpu_count()
    start = time.time()
    files_done = pd.DataFrame(glob.glob(output_path + job_id + '/*.csv'),columns = ['full_path'])
    files_done['file_name'] = files_done.full_path.str.split(output_path + job_id + '/result_job_' ,n=1).str[-1]
    start = time.time()
    # compteur
    cpt = 0
    while selected_files.shape[0]!= files_done.shape[0] and cpt < 10:
        # update compteur
        cpt += 1
        files_to_do = get_file_to_do(selected_files, files_done)
        files_to_do = files_to_do.drop(['index'],axis = 1)
        files_to_do = files_to_do.reset_index()
        if files_to_do.shape[0]<batch_size:
            batch_size = files_to_do.shape[0]
        
        for i in range(0,len(files_to_do),batch_size):
        # execute all tasks in a batch
            processes = [Process(target=task, args=(j,files_to_do,application_param, reservoir_param, output_path,job_id)) for j in range(i, i+batch_size) if j <=(len(files_to_do)-1)]
        # start all processes
            for process in processes:
                process.start()
        # wait for all processes to complete
            for process in processes:
                process.join()
        # report that all tasks are completed
        
        files_done = pd.DataFrame(glob.glob(output_path + job_id + '/*.csv'),columns = ['full_path'])
        files_done['file_name'] = files_done.full_path.str.split(output_path + job_id + '/result_job_' ,n=1).str[-1]
    # print('Done', flush=True)
    stop = time.time()
    # print(stop-start)
    perf = get_relative_baseline(output_path + job_id)
    return perf

# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# if __name__ == '__main__':
#     ## Get file list
#     path = '/home/tferte/predictcovid_api_python/data/'
#     #path = '/home/ddutartr/Projet/SISTM/ReservoirSistm/data/precompute_smoothing_csv/'
#     application_param = appParam()
#     reservoir_param = reservoirParam()
#     job_id = 'slurmtoto'
#     output_path ='/home/tferte/predictcovid_api_python/extdata/'
#     #output_path = '/home/ddutartr/Projet/SISTM/ReservoirSistm/extdata/'
#     
#     perf = perform_full_training(path, application_param, reservoir_param, job_id,output_path)
#     # print(perf)
