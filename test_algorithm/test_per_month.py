#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:03:34 2023

@author: ddutartr
"""
import pandas as pd
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def get_prediction_per_month( path , min_date_eval , nb_esn , top_trial):
    min_date_eval = datetime.strptime(min_date_eval, '%Y-%m-%d') - timedelta(days= 14)
    max_date_eval = pd.to_datetime(min_date_eval + relativedelta(months = 1))
    min_date_eval =pd.to_datetime(min_date_eval)

    top_trial_pred = pd.DataFrame(columns = ['outcomeDate', 'outcome', 'hosp', 'pred', 'nbFeatures', 'model',
               'mintraining','id_trial'])
    for i in range(0,top_trial):
        files = pd.DataFrame(glob.glob(path + str(i) + '/*.csv'),columns = ['full_path'])
        files['file_name'] = files.full_path.str.split(path + str(i),n=1).str[-1].str.split('_').str[-1]
       

        files['date'] = pd.to_datetime(files.file_name.str.split('.csv').str[0],format='%Y%m%d')
        files = files.sort_values(by='date').reset_index()
        

        # Selection by date
        selected_files = files[(files['date'] >= min_date_eval) & (files['date'] < max_date_eval) ]
        prediction = pd.DataFrame(columns = ['outcomeDate', 'outcome', 'hosp', 'pred', 'nbFeatures', 'model',
               'mintraining'])
        for file_name in (selected_files['full_path']):
            prediction = prediction.append(pd.read_csv(file_name)[:nb_esn])
        prediction['id_trial'] = i
        top_trial_pred = top_trial_pred.append(prediction)
        

    top_trial_pred.to_csv( path + 'pred_top_' + str(top_trial) + '_trial_nbesn' + str(nb_esn) + '.csv')
    return top_trial_pred
        
