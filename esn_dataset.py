#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:58:23 2024

@author: ddutartr
"""

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Electricity Transformer Temperature (ETT) dataset."""
from dataclasses import dataclass

import pandas as pd
import glob
from datetime import datetime,timedelta

import datasets
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "h1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "h2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "m1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "m2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

_CITATION = """\

"""

_DESCRIPTION = """\
The data ....
"""

_HOMEPAGE = ""

_LICENSE = ""

@dataclass
class ESNCovidConfig(datasets.BuilderConfig):
    """ESNCovidConfig builder config."""
    freq: str = "1D"
    prediction_length: int = 14
    min_date_eval: str = '2021-03-01'
    data_dir : str = 'data_obfuscated_time/'
    forecast_days : int = 14
    minanteriorite : int = 1000
    is_epidemio : bool = True
    is_best_param : bool = True


class ESNCovidDataset(datasets.GeneratorBasedBuilder):
    """ESNCovidDataset dataset"""

    VERSION = datasets.Version("1.0.0")

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('ett', 'h1')
    # data = datasets.load_dataset('ett', 'm2')
    BUILDER_CONFIGS = [
        ESNCovidConfig(
            name="esncovid",
            version=VERSION,
            description="Time series from XXXX.",
        )
    ]

    DEFAULT_CONFIG_NAME = "esncovid"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    
    def _info(self):
        features = datasets.Features(
            {
                "start": datasets.Value("timestamp[s]"),
                "target": datasets.Sequence(datasets.Value("float32")),
                "feat_static_cat": datasets.Sequence(datasets.Value("uint64")),
                "feat_dynamic_real":  datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                "item_id": datasets.Value("string"),
                "hosp" : datasets.Value("float32")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        print(self.config)
        #urls = _URLS[self.config.name]
        filepath = self.config.data_dir
        print(filepath)

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": filepath,
            #         "split": "dev",
            #     },
            # ),
            datasets.SplitGenerator(
                 name=datasets.Split.VALIDATION,
                 # These kwargs will be passed to _generate_examples
                 gen_kwargs={
                     "filepath": filepath,
                     "split": "val",
                 },
             ),
            datasets.SplitGenerator(
                 name=datasets.Split.TEST,
                 # These kwargs will be passed to _generate_examples
                 gen_kwargs={
                    "filepath": filepath,
                     "split": "test",
                },
             ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        
        files = pd.DataFrame(glob.glob(self.config.data_dir + '/*.csv'),columns = ['full_path'])
        files['file_name'] = files.full_path.str.split('/',n=3).str[-1]
        files['date'] = pd.to_datetime(files.file_name.str.split('.csv').str[0],format='%Y%m%d')
        files = files.sort_values(by='date').reset_index()
        min_date_eval = datetime.strptime(self.config.min_date_eval, '%Y-%m-%d') + timedelta(days=self.config.forecast_days)
        if split in ['dev','val']:
            if split == 'dev':
                selected_files= files[files['date']<min_date_eval]
                for i, path in enumerate(selected_files.full_path):
                    input_data = pd.read_csv(path)
                    df = input_data.copy()
                    
                    df = df[df['outcomeDate'] <= max(df['START_DATE'])]

                    df.outcome= df.loc[:,"outcome"].values-df.loc[:,"hosp"].values
                    df = df.drop(['START_DATE'],axis=1)
                    
                    df = df.tail( n = self.config.minanteriorite )
                    df = df.reset_index()
                    start = datetime.strptime(df['outcomeDate'][0],'%Y-%m-%d')
                    target = df.outcome.values.astype("float32").T
                    item_id = files['file_name'][i]
                    hosp =  df.hosp.values.astype("float32").T[-1]
                    if self.config.is_epidemio:
                        vecFeaturesEpi = ["hosp", "hosp_rolDeriv7",
                       "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                       "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                       "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                       "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                       "IPTCC.mean",
                       "Vaccin_1dose",
                       "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
                        if self.config.is_best_param:
                            files_features = "/home/tf1/Documents/recherche/prediction_covid/high_dimension_reservoir/results/best_features.csv"
                            feat= pd.read_csv(files_features)
                            feat = feat[feat['last_used_observation'] == "2021-03-01"] 
                            vecFeaturesEpi = feat.name.tolist()
                        column_names = df.columns.tolist()
                        vecFeatures = [c for c in vecFeaturesEpi if c in column_names]
                        feat_dynamic_real = df[vecFeatures].values.T.astype("float32")
                        yield i, {
                            "start": start,
                            "target": target,
                            "feat_dynamic_real": feat_dynamic_real,
                            "feat_static_cat": [0],
                            "item_id": item_id,
                            "hosp" : hosp
                        }
            else:
                selected_files = files[files['date']<min_date_eval]
                selected_files = selected_files.iloc[selected_files.shape[0]-1]
                input_data = pd.read_csv(selected_files.full_path)
                df = input_data.copy()
                df.outcome= df.loc[:,"outcome"].values-df.loc[:,"hosp"].values
                df = df.drop(['START_DATE'],axis=1)
                df = df.tail( n = self.config.minanteriorite )
                df = df.reset_index()
                start = datetime.strptime(df['outcomeDate'][0],'%Y-%m-%d')
                target = df.outcome.values.astype("float32").T
                item_id = selected_files['file_name']
                hosp =  df.hosp.values.astype("float32").T[-1]
                if self.config.is_epidemio:
                    vecFeaturesEpi = ["hosp", "hosp_rolDeriv7",
                   "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
                   "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
                   "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
                   "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
                   "IPTCC.mean",
                   "Vaccin_1dose",
                   "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
                    if self.config.is_best_param:
                        files_features = "/home/tf1/Documents/recherche/prediction_covid/high_dimension_reservoir/results/best_features.csv"
                        feat= pd.read_csv(files_features)
                        feat = feat[feat['last_used_observation'] == "2021-03-01"] 
                        vecFeaturesEpi = feat.name.tolist()
                    column_names = df.columns.tolist()
                    vecFeatures = [c for c in vecFeaturesEpi if c in column_names]
                    feat_dynamic_real = df[vecFeatures].values.T.astype("float32")
                    yield 0, {
                        "start": start,
                        "target": target,
                        "feat_dynamic_real": feat_dynamic_real,
                        "feat_static_cat": [0],
                        "item_id": item_id,
                        "hosp" : hosp
                    }
        if split == 'test':
            selected_files = files[files['date']<min_date_eval]
            selected_files = selected_files.iloc[selected_files.shape[0]-1]
            input_data = pd.read_csv(selected_files.full_path)
            df = input_data.copy()
            
            df = df[df['outcomeDate'] <= max(df['START_DATE'])]
            
            
            df.outcome= df.loc[:,"outcome"].values-df.loc[:,"hosp"].values
            
            
            df = df.drop(['START_DATE'],axis=1)
            df = df.tail( n = self.config.minanteriorite )
            df = df.reset_index()

            start = datetime.strptime(df['outcomeDate'][0],'%Y-%m-%d')
            target = df.outcome.values.astype("float32").T
            hosp =  df.hosp.values.astype("float32").T[-1]
            item_id = selected_files['file_name']
            if self.config.is_epidemio:
                vecFeaturesEpi = ["hosp", "hosp_rolDeriv7",
               "P_TOUS_AGES", "P_TOUS_AGES_rolDeriv7",
               "P_60_90_PLUS_ANS", "P_60_90_PLUS_ANS_rolDeriv7",
               "FRACP_TOUS_AGES", "FRACP_TOUS_AGES_rolDeriv7",
               "FRACP_60_90_PLUS_ANS", "FRACP_60_90_PLUS_ANS_rolDeriv7",
               "IPTCC.mean",
               "Vaccin_1dose",
               "URG_covid_19_COUNT", "URG_covid_19_COUNT_rolDeriv7"]
                if self.config.is_best_param:
                    files_features = "/home/tf1/Documents/recherche/prediction_covid/high_dimension_reservoir/results/best_features.csv"
                    feat= pd.read_csv(files_features)
                    feat = feat[feat['last_used_observation'] == "2021-03-01"] 
                    vecFeaturesEpi = feat.name.tolist()
                column_names = df.columns.tolist()
                vecFeatures = [c for c in vecFeaturesEpi if c in column_names]
                feat_dynamic_real = df[vecFeatures].values.T.astype("float32")
                yield 0, {
                    "start": start,
                    "target": target,
                    "feat_dynamic_real": feat_dynamic_real,
                    "feat_static_cat": [0],
                    "item_id": item_id,
                    "hosp" : hosp
                }
                
            
                

        
