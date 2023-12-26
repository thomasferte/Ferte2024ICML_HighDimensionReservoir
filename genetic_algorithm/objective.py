#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:52:52 2023

@author: ddutartr
"""
from train_test_api.utils import *


def objective_epidemio_common_is(trial,min_date_eval='2021-03-01'):
    ridge = trial.suggest_float("ridge", 1e-10, 1e5, log=True)
    spectral_radius = trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True)
    input_scaling = trial.suggest_float("input_scaling", 1e-5, 1e5, log=True)
    leaking_rate = trial.suggest_float("leaking_rate", 1e-5, 1, log=True) 

    fct_value = perform_full_training(
      path='/home/ddutartr/Projet/SISTM/Genetics/predictcovid_api_python/data_short/',
      application_param=appParam(mintraining=1000, nb_esn= 3, is_training=True),
      reservoir_param=reservoirParam(
        units=500,
        ridge=ridge,
        input_scaling=input_scaling,
        leaking_rate=leaking_rate,
        spectral_radius=spectral_radius),
      job_id=str(trial.number),
      output_path="./simulations/01_epidemio_common_is/output/" + min_date_eval + '/',
      min_date_eval = min_date_eval)
    
    return fct_value

def objective_epidemio_multiple_is(trial):
    columns_to_keep = ["hosp",
    "hosp_rolDeriv7",
    "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
    "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
    "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
    "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
    "IPTCC.mean",
    "Vaccin_1dose",
    "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
    ridge = trial.suggest_float("ridge", 1e-10, 1e5, log=True)
    spectral_radius = trial.suggest_float("spectral_radius", 1e-5, 1e5, log=True)
    leaking_rate = trial.suggest_float("leaking_rate", 1e-5, 1, log=True) 
    
    input_scaling = {}
    for column in columns_to_keep:
        suggested_value = trial.suggest_float(column, 1e-5, 1e5, log=True)
        input_scaling[column] = suggested_value

    fct_value = perform_full_training(
      path="/home/ddutartr/Projet/SISTM/ReservoirSistm/data/precompute_smoothing_csv/",
      application_param=appParam(mintraining=1000, nb_esn= 3, is_training=True),
      reservoir_param=reservoirParam(
        units=500,
        ridge=ridge,
        input_scaling=input_scaling,
        leaking_rate=leaking_rate,
        spectral_radius=spectral_radius),
      job_id=str(trial.number),
      output_path="./simulations/02_epidemio_multiple_is/output/")
    
    return fct_value