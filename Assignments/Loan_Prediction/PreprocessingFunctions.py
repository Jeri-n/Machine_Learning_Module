# loading libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




#missing value imputation - 
def missing_value_imputation(data,fill_value,fill_type,columns):
    print('Missing values before Treatment is \n',data.isna().sum())

    for column in columns:

        if 'median replacement' in fill_type:
            median=data[column].median()
            data[column].fillna(median,inplace=True)

        # if 'Random sample imputation' in fill_type:
        #     random_samples=data[column].dropna().sample(data[column].isna().sum(),random_state=0,replace=True)
        #     random_samples.index=data[data[column].isna()].index
        #     data.loc[data[column].isnull()] =random_samples	
        
        # if "Random_sample_Fill" in fill_type:
        #     data[column+"_random"]=data[column]
        #         ##It will have the random sample to fill the na
        #     random_sample = data[column].dropna().sample(data[column].isnull().sum(),random_state=0)
        #         ##pandas need to have same index in order to merge the dataset
        #     random_sample.index=data[data[column].isnull()].index
        #     data.loc[data[column].isnull(),column+'_random']=random_sample
        #     data[column]=data[column+"_random"]
        #     data.drop([column+"_random"],axis=1,inplace=True)
        
        if "Random_sample_Fill" in fill_type:
            data[column+"_random"]=data[column]
            ##It will have the random sample to fill the na
            random_sample = data[column].dropna().sample(data[column].isnull().sum(),random_state=0)
            ##pandas need to have same index in order to merge the dataset
            random_sample.index=data[data[column].isnull()].index
            data.loc[data[column].isnull(),column+'_random']=random_sample
            data[column]=data[column+"_random"]
            data.drop([column+"_random"],axis=1,inplace=True)

        if 'New feature importance' in fill_type:
            np.where(data[column].isna(),1,0,inplace=True)
            data[column].fillna(median,inplace=True)


        if 'mode fill' in fill_type:
            mod=data[column].mode()[0]
            data[column].fillna(mod,inplace=True)

        if 'End of distribution imputation' in fill_type:
            extreme = data[column].mean()+3*(data[column].std())
            data[column].fillna(extreme,inplace=True)
            

        if 'value fill' in fill_type:
            data[column].fillna(fill_value,inplace=True)

    print('\n Missing values After Treatment is \n',data.isna().sum())

    return(data)

    
