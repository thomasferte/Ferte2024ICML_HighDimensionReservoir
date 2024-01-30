import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
import glob
import time



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)

        x,_ = self.lstm(x,(h0,c0))
        x = self.fc(x[:,-1,:])
        return x
        
def get_number_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    return params

class appParam(object):
    def __init__(self, vecFeaturesEpi = ["hosp", "hosp_rolDeriv7",
                   "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                   "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                   "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                   "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                   "IPTCC.mean",
                   "Vaccin_1dose",
                   "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"],
                mintraining = 335 , warmup = 0 , nb_esn = 10, doEnet = False , beta_linear_enet_is = 0, beta_sqrt_enet_is = 0, name = None, is_training = True):
        self.mintraining = mintraining
        self.warmup = warmup
        self.minanteriorite = self.warmup + self.mintraining
        self.vecFeaturesEpi = vecFeaturesEpi
        self.name = name
        self.is_training = is_training
        
        # check vecfeature type
        if not isinstance(vecFeaturesEpi, list): 
            raise Exception("vecFeaturesEpi must be a list (e.g ['all']")
        if len(vecFeaturesEpi) == 1 and vecFeaturesEpi[0] != "all": 
            raise Warning("If vecFeaturesEpi is of length 1, it must be equal to ['all'] to select all features")
            
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

def do_pca(X,pca_value = 0.7,pca=None):
    if pca is None:
        pca = PCA(n_components=30)
        pca.fit(X)
        n_components, = np.where(np.cumsum(pca.explained_variance_ratio_)>pca_value)
        if len(n_components)==0:
            n_components=[1]
        pca = PCA(n_components=n_components[0])
        X_r = pca.fit(X).transform(X)
    else:
        X_r = pca.transform(X)
    return X_r,pca


def select_features_from_names(df, application_param):
    """Select features on dataframe based on list of names
    Parameters
    ----------
    df : dataframe
        Raw data
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
    
    # select the columns from dataframe
    df_selected= df[vecFeatures]
    df_selected = df[vecFeatures]
    # remove constant features
    df_selected = df_selected.loc[:,df_selected.apply(pd.Series.nunique) != 1]
    df_selected['outcome'] = df.outcome
    df_selected['outcomeDate'] = df.outcomeDate
    
    # feat_dynamic_real = df[vecFeatures].values
    # a,pca = do_pca(feat_dynamic_real)
    # feat_dynamic_real=a.T.astype("float32")
    # X = np.array(feat_dynamic_real)
    # a,pca = do_pca(feat_dynamic_real)
    
    return df_selected


def get_relative_baseline(path_out):
    files_result = glob.glob(path_out + '/*.csv')
    
    if len(files_result) == 0 :
        return 1000
    
    temp = []

    for filename in files_result:
        df = pd.read_csv(filename, index_col=None, header=0)
        temp.append(df)

    df_result = pd.concat(temp, axis=0, ignore_index=True)
    df_result.loc[df_result['pred']<10,'pred'] = 10
    df_result.loc[df_result['outcome']<10,'outcome'] = 10
    df_result.loc[df_result['hosp']<10,'hosp'] = 10
    df_grouped_by_date = df_result.groupby(['outcomeDate', 'model']).median()[['outcome','hosp','pred']]
    df_filter = df_grouped_by_date.loc[~(df_grouped_by_date['outcome'].isna())].copy()
    df_filter['AE'] = np.abs(df_filter['pred']-df_filter['outcome'])
    df_filter['AE_baseline'] = np.abs(df_filter['hosp']-df_filter['outcome'])
    df_filter['delta_baseline'] = df_filter['AE']-df_filter['AE_baseline']
    df_filter['relative_baseline']=df_filter['AE']/df_filter['AE_baseline']
    df_filter['RE'] = df_filter['AE']/df_filter['outcome']

    return df_filter['AE'].mean()

from sklearn.decomposition import PCA
def do_pca(X,pca_value = 0.7,pca=None):
    if pca is None:
        pca = PCA(n_components=30)
        pca.fit(X)
        n_components, = np.where(np.cumsum(pca.explained_variance_ratio_)>pca_value)
        if len(n_components)==0:
            n_components=[1]
        pca = PCA(n_components=n_components[0])
        X_r = pca.fit(X).transform(X)
    else:
        X_r = pca.transform(X)
    return X_r,pca
def pref_on_test_set(dftest, selected_columns , application_param , norm_array, model, n_seq,pca,percent_pca):
    dftest_selected = dftest[selected_columns]
    X_esn, Y_esn, scaling = standardise_data_for_ens(dftest_selected, norm_array )
    X_esn, pca = do_pca(X_esn,percent_pca,pca)
    X_esn, Y_esn = create_seq(X_esn,Y_esn,n_seq = n_seq)
    X_test = torch.from_numpy(X_esn).float()
    vecPred = model(X_test).squeeze().detach().numpy()
    dfres = dftest.copy().reset_index()
    dfres = dfres.loc[n_seq+1:]
    dfres['pred'] = np.squeeze(vecPred)
    dfres['pred'] = np.squeeze(vecPred) + dfres['hosp']
    dfres = dfres[['outcomeDate','outcome','hosp','pred']].tail(1)
    dfres['nbFeatures'] = np.shape(X_esn)[2]
    dfres['model'] = "LSTM"
    dfres['mintraining'] = application_param.mintraining
    return dfres


def create_seq(X,Y,n_seq):
    xs , ys = [] , []
    for i in range(len(X)-n_seq-1):
        x = X[i:(i+n_seq)]
        y= Y[i+n_seq]
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_lstm(selected_files,index,application_param,output_path,job_id,learning_rate=1e-3,num_epochs=2000,n_seq=3,ld=1e-4,percent_pca=0.7):   
    file_i = selected_files.reset_index().full_path[index]
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
    df_select = select_features_from_names(df=dftrain , application_param=application_param)
    selected_columns = df_select.columns
    # Do normalisation for ESN
    X_esn, Y_esn, scaling = standardise_data_for_ens(df=df_select)
    X_esn,pca = do_pca(X_esn,percent_pca)
    X_esn, Y_esn = create_seq(X_esn,Y_esn,n_seq = n_seq)
    X_train,X_test = X_esn[:-7],X_esn[-7:]
    Y_train,Y_test = Y_esn[:-7],Y_esn[-7:]
    print(X_esn.shape)
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).float()
    
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    import random
    norm_array = scaling['vecMaxAbs']
    #pred_esn = pd.DataFrame(columns = ["outcomeDate", "outcome", "hosp","pred","nbFeatures","model", "mintraining","lr","n_epoch","n_seq"])
    number = np.random.randint(100)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.backends.cudnn.deterministic = True
    number
    model = LSTM(X_train.shape[-1],hidden_size=6,num_layers=1,output_size=Y_train.shape[-1])
    
    l2_regularisation = ld

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.95) ,weight_decay=l2_regularisation)
    criterion = nn.MSELoss()
    
    early_stopper = EarlyStopper(patience=5, min_delta=3)
    train_loss = []
    validation_loss = []
    for epoch in range(num_epochs):
        outputs = model(X_train).squeeze()
        optimizer.zero_grad()
        loss = criterion(outputs,Y_train.squeeze())
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().numpy())
        with torch.no_grad():
            outputs = model(X_test).squeeze()
            validate_loss =  criterion(outputs,Y_test.squeeze())
            validation_loss.append(validate_loss)
            if epoch > 250:
                if early_stopper.early_stop(validate_loss):             
                    print("We are at epoch:", epoch)
                    break
    epoch_max = epoch
    X_train = torch.from_numpy(X_esn).float()
    Y_train = torch.from_numpy(Y_esn).float()
    model = LSTM(X_train.shape[-1],hidden_size=6,num_layers=1,output_size=Y_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularisation)
    for epoch in range(epoch_max):
        outputs = model(X_train).squeeze()
        optimizer.zero_grad()
        loss = criterion(outputs,Y_train.squeeze())
        loss.backward()
        optimizer.step()
    pred_j_esn = pref_on_test_set(dftest=dftest,
      selected_columns = selected_columns,
      application_param=application_param ,
      norm_array=norm_array,
      model=model,n_seq = n_seq,pca=pca,percent_pca = percent_pca)
    pred_j_esn["lr"]= learning_rate
    pred_j_esn["n_epoch"] = num_epochs
    pred_j_esn["n_seq"]= n_seq
    pred_j_esn["epoch"]= epoch
    pred_j_esn["l2"]= l2_regularisation
    pred_j_esn["percent_pca"]= percent_pca
    
    pred_esn = pred_j_esn.reset_index()
    
    pred_esn.to_csv(output_path+ job_id +'/result_job_'+ selected_files.reset_index().file_name[index],index=False)




def create_params():
    lr = np.random.uniform(low=1e-4, high=0.01)
    n_seq = np.random.randint(1, 3)
    num_epoch = np.random.randint(500, 2200)
    
    lr, num_epoch,n_seq = 0.00955059115857072 ,  920, 5
    ld = np.random.uniform(low=1e-7, high=0.01)
    percent_pca = np.random.randint(30, 99)/100

    return (lr, n_seq, num_epoch,ld,percent_pca)


def perform_full_training(path, application_param , job_id,output_path, best_param = None,min_date_eval='2021-03-01', forecast_days=14):
    files = pd.DataFrame(glob.glob(path + '*.csv'),columns = ['full_path'])
    print(files)
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
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(output_path + job_id, exist_ok = True)
    if best_param is None :
        lr, n_seq, num_epoch,ld,percent_pca = create_params()
    else:
        lr,num_epoch, n_seq,ld,percent_pca = best_param
    # Loop over file
    for i in range(selected_files.shape[0]):
        print(i)
        train_lstm(selected_files,i,application_param,output_path,job_id,lr,num_epoch,n_seq,ld,percent_pca)
    perf = get_relative_baseline(output_path + job_id)
    # files = glob.glob(output_path + job_id + '/*')
    # for f in files:
    #     os.remove(f)
        
    return perf , lr , num_epoch , n_seq,ld,percent_pca


if __name__ == '__main__':

    # path = '/home/ddutartr/Projet/SISTM/LSTM/data/data_obfuscated/'
    # path = '/beegfs/ddutartr/LSTM/data_obfuscated/'
    path = "data_obfuscated_time/"
    application_param = appParam()
    application_param.vecFeaturesEpi = ['all']
    job_id = 'slurmtoto_PCA_for_thomas'
    # output_path = '/home/ddutartr/Projet/SISTM/LSTM/output/'
    # output_path = '/beegfs/ddutartr/LSTM/output/'
    output_path = 'output/LSTM/'
    # log_file = "/home/ddutartr/Projet/SISTM/LSTM/log.csv"
    log_file = "output/LSTMlog.csv"
    application_param.is_training = False
    # log_file = "/beegfs/ddutartr/LSTM/log2.csv"
    best_param = 0.00955059115857072 ,  920, 5,3.89884610e-03, 0.87
    
    start_time = time.time()
    
    perf , lr , num_epoch, n_seq,ld,percent_pca = perform_full_training(path, application_param, job_id,output_path,best_param)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    results = {
        'Scenario': "LSTM",
        'Elapsed Time': [elapsed_time]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/timing_LSTM.csv")
        
    perf  
