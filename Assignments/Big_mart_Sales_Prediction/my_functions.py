#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # forloop for checking missing vales col by col

# # for features in df.columns:
# #     if df[features].isnull().sum()>0:
# #         print("The column name",features,"has",df[features].isnull().sum(),"missing values")


# def missing_value_imputation(data,fill_value,fill_type,columns):
#     print('Missing values before Treatment is \n',data.isna().sum())

#     for column in columns:

#         if 'median replacement' in fill_type:
#             median=data[column].median()
#             data[column].fillna(median,inplace=True)

#         if 'Random sample imputation' in fill_type:
#             random_samples=data[column].dropna().sample(data[column].isna().sum(),random_state=0,replace=True)
#             random_samples.index=data[data[column].isna()].index
#             data.loc[data[column].isnull()] =random_samples	


#         if 'New feature importance' in fill_type:
#             np.where(data[column].isna(),1,0,inplace=True)
#             data[column].fillna(median,inplace=True)


#         if 'mode_fill' in fill_type:
#             mod=data[column].mode()[0]
#             data[column].fillna(mod,inplace=True)

#         if 'End of distribution imputation' in fill_type:
#             extreme = data[column].mean()+3*(data[column].std())
#             data[column].fillna(extreme,inplace=True)
            

#         if 'value fill' in fill_type:
#             data[column].fillna(fill_value,inplace=True)

#     print('Missing values After Treatment is \n',data.isna().sum())

#     return(data)




  # checking for outliers - boxplots


  
def boxplots(data):
    for feature in data.columns:
        plt.figure(figsize=(10,1))
        if data[feature].dtype !='object':
            sns.boxplot(data[feature],data=data)
        else:
            print(feature,'is an object')

 