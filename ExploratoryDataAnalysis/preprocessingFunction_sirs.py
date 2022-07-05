import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Python Method 1 : Displays Data Information :
#=============================================#

def display_data_information(data,data_types,dataframe_name):
    print("Information of ",dataframe_name,": Rows = ",data.shape[0], "| Columns = ",data.shape[1],"\n")
    data.info()
    print("\n")
    for variable in data_types:
        data_type = data.select_dtypes(include=[variable]).dtypes
        if len(data_type) > 0:
            #print(str(len(data_type))+" "+variable +" Features\n"+str(data_type)+"\n")
            print(str(len(data_type))+" "+variable +" Features")
# data_types  = ["float32","float64","int32","int64","object","category","datetime64[ns]"]
# display_data_information(train,data_types,"train")

# Python Method 2 : Displays Data Head (Top Rows) and Tail (Bottom Rows) of the Dataframe (Table) :
#=================================================================================================#
def display_head_tail(data,head_rows,tail_rows):
    display("Data Head and Tail : ")
    display(data.head(head_rows).append(data.tail(tail_rows)))

# Displays Data Head (Top Rows) and Tail (Bottom Rows) of the Dataframe (Table)

# display_head_tail(train, head_rows=3, tail_rows=2)

# Python Method 3 : Displays Data Description using Statistics :
#==============================================================#
def data_features_dtypes(data):
    numeric_features = data.select_dtypes(include=[np.number])
    categorical_features = data.select_dtypes(include=[np.object])
    return numeric_features,categorical_features

def display_data_description(data):
    print("Data Description : ")
    display(data.describe(include=[np.number]))
    numeric_features = data.select_dtypes(include=[np.number])
    print("The Identified Numeric Features are : \n", numeric_features.columns)
    numericalColumns = numeric_features.columns
    print(" ")
    display(data.describe(include=[np.object]))
    categorical_features = data.select_dtypes(include=[np.object])
    print("The Identified Caterogical Features are :\n", categorical_features.columns)
    categoricalColumns = categorical_features.columns
    return numericalColumns , categoricalColumns
# Display Data Description of "Train/test" :

# numeric_features,categorical_features = display_data_description(train)

# Python Method 4 : Removes Data Duplicates while Retaining the First one - Similar to SQL DISTINCT :
#===================================================================================================#
def remove_duplicates(data):
    print("BEFORE REMOVING DUPLICATES - No. of Rows = ",data.shape[0])
    data.drop_duplicates(keep="first",inplace=True)
    print("AFTER REMOVING DUPLICATES  - No. of Rows = ",data.shape[0])
    return data

# Remove Duplicates from "train" data :
# train = remove_duplicates(train)

# Python Method 5 : Displays Unique Values in Each Column of the Dataframe(Table) :
#=================================================================================#
def display_unique(data,feature_dtype):

    numeric_features,categorical_features = data_features_dtypes(data)
    if feature_dtype=="categorical":
        columns = categorical_features.columns
        print("Total Number of"+" categorical is",len(columns))
    elif feature_dtype=="numerical":
        columns = numeric_features.columns
        print("Total Number of"+" numerical is",len(columns))
    else:
        print("Feature Dtype is not supported!!!")

    

    for column in columns:
        print("No of Unique Values in "+column+" Column are : "+str(data[column].nunique()))
        
# display_unique(train,"numerical")
# display_unique(train,"categorical")

# Python Method 7 : identify the Missing values features, for numerical and categoricalfeatures
#=============================================================================================#
def missing_Value_features(data):
    moreNullNum=[]
    nullNum=[]
    moreNullCat=[]
    nullCat=[]
    numeric_features,categorical_features = data_features_dtypes(data)
    num_features_with_na= [ feature for feature in numeric_features.columns if numeric_features[feature].isnull().sum()>0]
    print("The Missing values found for Numerical Data Types :\n")
    for feature in num_features_with_na:
        print(feature, np.round(100*numeric_features[feature].isnull().sum()/data.shape[0],4),"% of Missing values found")
        if np.round(100*numeric_features[feature].isnull().sum()/data.shape[0],4) >=50:
            moreNullNum.append(feature)
        else:
            nullNum.append(feature)
    print("\n")
    print("The Missing values found for Categorical Data Types :\n")
    cat_features_with_na= [ feature for feature in categorical_features.columns if categorical_features[feature].isnull().sum()>0]
    for feature in cat_features_with_na:
        print(feature, np.round(100*categorical_features[feature].isnull().sum()/data.shape[0],4),"% of Missing values found")
        if np.round(100*categorical_features[feature].isnull().sum()/data.shape[0],4) >= 50:
            moreNullCat.append(feature)
        else:
            nullCat.append(feature)
    
    print("\n Features having more  Missing value found for the Numerical Datatype:\n",moreNullNum)
    print("Features having more  Missing value found for the Categorical Datatype:\n",moreNullCat)

    print("Features having less Missing value found for the Numerical Datatype:\n",nullNum)
    print("Features having less Missing value found for the Categorical Datatype:\n",nullCat)

    return nullNum,nullCat
    
# nullNum,nullCat = missing_Value_features(train)

# Python Method 7 : Fills or Imputes Missing values with Various Methods :
#========================================================================#

def fill_missing_values(data,fill_value,fill_types,columns,dataframe_name):

    

    print("Missing Values BEFORE REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())

    for column in columns:
        
        if "Random_sample_Fill" in fill_types:
            data[column+"_random"]=data[column]
            ##It will have the random sample to fill the na
            random_sample = data[column].dropna().sample(data[column].isnull().sum(),random_state=0)
            ##pandas need to have same index in order to merge the dataset
            random_sample.index=data[data[column].isnull()].index
            data.loc[data[column].isnull(),column+'_random']=random_sample
            data[column]=data[column+"_random"]
            data.drop([column+"_random"],axis=1,inplace=True)

        if "New_Feature_Importance" in fill_types :
            data[column+'_NAN'] = np.where(data[column].isnull(),1,0)
            data[column].fillna(data[column].median(),inplace=True)

        # Fill missing values with median values: --- > For Numeric features
        if "Median_Fill" in fill_types :
            data[column].fillna(data[column].median(),inplace=True)

        # Fill missing values with Mode values: --- > For Categorical features
        if "Mode_Fill" in fill_types :
            data[column].fillna(data[column].mode()[0],inplace=True)

        # Fill missing values with Specific values: --- > For Numeric/Categorical features
        if "Value_Fill" in fill_types :
            data[column].fillna(fill_value,inplace=True)

        # Fill Missing Values with Forward Fill  (Previous Row Value as Current Row in Table) : --- > For Numeric/Categorical features
        if "Forward_Fill" in fill_types :
            data[column].ffill(axis = 0, inplace=True)

        # Fill Missing Values with Backward Fill (Next Row Value as Current Row in Table) : --- > For Numeric/Categorical features
        if "Backward_Fill" in fill_types :
            data[ column ] = data[ column ].bfill(axis = 0)

    print("Missing Values AFTER REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())

        
    return data

# Python Method 8 : Outlier Caping :
#========================================#
def Outlier_Detection(data,column,method,capValue=False):
    if "SD" in method:
        #Dropping the outlier rows with standard deviation
        factor = 3
        upper_lim = data[column].mean () + data[column].std () * factor
        lower_lim = data[column].mean () - data[column].std () * factor
        print(f"The upper limit for {column} is found to be {upper_lim}.")
        print(f"The lower limit for {column} is found to be {lower_lim}.")

        data = data[(data[column] < upper_lim) & (data[column] > lower_lim)]

        if capValue==True:
            data.loc[(data[column] > upper_lim),column] = upper_lim
            data.loc[(data[column] < lower_lim),column] = lower_lim
    if "Percentile" in method:
        upper_lim = data[column].quantile(.95)
        lower_lim = data[column].quantile(.05)
        data = data[(data[column] < upper_lim) & (data[column] > lower_lim)]
        print(f"The upper limit for {column} is found to be {upper_lim}.")
        print(f"The lower limit for {column} is found to be {lower_lim}.")

        if capValue==True:
            data.loc[(data[column] > upper_lim),column] = upper_lim
            data.loc[(data[column] < lower_lim),column] = lower_lim

    return data

# Python Method 9 : Categorical Encoding :
#========================================#

def categorical_encoding(data,columns,targetColumn,encodingType,approach):

    if encodingType=="NominalEncoding":
        if "OneHotEncoding" in approach:
            for column in columns:
                data_new = pd.get_dummies(data[column],drop_first=True)
                data = pd.concat([data,data_new],axis=1)
                data.drop(column,axis=1,inplace=True)
        if "OneHotEncodingWithManycatergories" in approach:
            for column in columns:
                lst_10=list(data[column].value_counts().sort_values(ascending=False).head(10).index)
                for categories in lst_10:
                    data[categories]=np.where(data[column]==categories,1,0)
        if "MeanEncoding" in approach:
            for column in columns:
                mean_ordinal=data.groupby([column])[targetColumn].mean().to_dict()
                data[column]=data[column].map(mean_ordinal)


    if encodingType=="OrdinalEncoding":
        if "LabelEncoding" in approach:
            le = LabelEncoder()
            data[columns] = data[columns].apply(le.fit_transform)

        if "TargetGuidedOrdinalEncoding" in approach:
            for column in columns:
                ordinal_labels=data.groupby([column])[targetColumn].mean().sort_values().index
                ordinal_labels={k:i for i,k in enumerate(ordinal_labels)}
                data[column] = data[column].map(ordinal_labels)

    return data

# Python Method 10 : Scaling Down :
#================================#
def scaling_data(data,scalingType):
    if "Standarization" in scalingType:
        scaler = StandardScaler()
        data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
    if "MinMaxs" in scalingType:
        scaler= MinMaxScaler()
        data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
    if "Robust" in scalingType:   
        scaler = RobustScaler()
        data= pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
    return data

