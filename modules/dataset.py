import pandas as pd
import numpy as np
from sklearn import preprocessing
df=pd.read_csv('1.csv')
column_number=df.shape[1]
df_columns_name=list(df.columns)
feature_dict=dict()

# missing value
def missing_value(df):
    for i in df.iloc:
        if i.isna().sum()!=0:
            i.dropna(inplace=True)
    return unique_features_value(df)
# veri setindeki id kolonu hariç featureları sözlüğe sütun sayılarına göre ekliyor.
def unique_features_value(df):
    # ilk süttun id olduğundan id'yi sözlüğe eklemiyoruz.
    for i in range(1,column_number):
        l = list()
        l=np.array(l)
        for j in df.iloc:
            l=np.append(l,j[i])
        feature_dict[i]=np.unique(l)
        l=[]
    return feature_dict,auto_encoding(df,feature_dict)

def auto_encoding(df,feature_dict):
    ohe=preprocessing.OneHotEncoder()
    le=preprocessing.LabelEncoder()
    dataframes=[]
    dataframes.append(df.iloc[:,0]) # id sütunu eklendi.
    # auto encoding
    for key,value in feature_dict.items():
        if len(value)>2:
            one_hot_encoding=df.iloc[:,key:key+1].values
            one_hot_encoding=ohe.fit_transform(one_hot_encoding).toarray()
            df_cols=pd.DataFrame(one_hot_encoding,columns=value,dtype='int32')
            dataframes.append(df_cols)
        else:
            binary_encoding=df.iloc[:,key:key+1].values
            binary_encoding[:,0]=le.fit_transform(df.iloc[:,key:key+1])
            df_cols=pd.DataFrame(binary_encoding,columns=[df_columns_name[key]],dtype='int32')
            dataframes.append(df_cols)
    # encoding haldeki data
    result_df=pd.concat(dataframes,axis=1)
    return result_df
unique_dict,result_df=missing_value(df)

events=['activity_01','activity_02','activity_03','activity_04','activity_05','*','activity_07']
def model_encoding(df,ac1,ac2,age,income,employed):
    activities=[ac1,ac2]
    arr=[]
    arr=np.array(arr)
    for i in df.iloc:
        if i[1]==age and i[2]==income and i[0]==employed:
            arr=np.append(arr,activities[1])
        else:
            arr=np.append(arr,activities[0])
    ohe=preprocessing.OneHotEncoder()
    le=preprocessing.LabelEncoder()
    df['target']=arr
    
    df['target']=[events.index(i)+1 if i in events else 0 for i in df['target'].iloc]
    #df['target']=np.where(df['target']==activities[0],0,1)
    target=df.iloc[:,3:4].values
    df_target=pd.DataFrame(target,columns=['target'])
    
    age=df.iloc[:,1:2].values
    age=ohe.fit_transform(age).toarray()
    df_age=pd.DataFrame(age,columns=['0-19yo','40-59yo','60-79yo','80yo+'],dtype='int32')
    
    income=df.iloc[:,2:3].values
    income=ohe.fit_transform(income).toarray()
    df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')
    
    employed=df.iloc[:,0:1].values
    employed[:,0]=le.fit_transform(df.iloc[:,0])
    df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')
    
    frames=[df_employed,df_age,df_income,df_target]
    df_result = pd.concat(frames,axis=1)
    return df_result

def model_multi_encoding(df,ac1,ac2,ac3,ac4,age,age2,income,income2,employed,employed2):
    activities=[ac1,ac2,ac3,ac4]
    arr=[]
    arr=np.array(arr)
    count=0
    for i in df.iloc:
        if i[1]==age and i[2]==income and i[0]==employed and count%2==0:
            arr=np.append(arr,activities[0])
        elif i[1]==age and i[2]==income and i[0]==employed and count%2!=0:
            arr=np.append(arr,activities[2])
        elif i[1]==age2 and i[2]==income2 and i[0]==employed2:
            arr=np.append(arr,activities[3])
        else:
            arr=np.append(arr,activities[1])
        count+=1
    ohe=preprocessing.OneHotEncoder()
    le=preprocessing.LabelEncoder()
    df['target']=arr
    
    df['target']=[events.index(i)+1 if i in events else 0 for i in df['target'].iloc]

    target=df.iloc[:,3:4].values
    #target[:,0]=le.fit_transform(df.iloc[:,3:4])
    df_target=pd.DataFrame(target,columns=['target'],dtype='int32')
    
    age=df.iloc[:,1:2].values
    age=ohe.fit_transform(age).toarray()
    df_age=pd.DataFrame(age,columns=['0-19yo','40-59yo','60-79yo','80yo+'],dtype='int32')
    
    income=df.iloc[:,2:3].values
    income=ohe.fit_transform(income).toarray()
    df_income=pd.DataFrame(income,columns=['high','low','middle'],dtype='int32')
    
    employed=df.iloc[:,0:1].values
    employed[:,0]=le.fit_transform(df.iloc[:,0])
    df_employed=pd.DataFrame(employed,columns=['employed'],dtype='int32')
    
    frames=[df_employed,df_age,df_income,df_target]
    df_result= pd.concat(frames,axis=1)
    return df_result
    
# korelasyon hesapla
def calculate_corr(df):
    corr=result_df.corr()
    return corr
corr=calculate_corr(result_df)
# yolculukları belirle
def find_journey(df):
    event=df_columns_name[1] # yinelenen eventler
    del df_columns_name[1] 
    # her yinelenen id için journeyler oluşturuluyor.
    df_with_journey = (df.groupby(df_columns_name)
      .agg({event: lambda x: x.tolist()})
      .reset_index())
    # journeylerin adetini ve tekilliğini bulmak için list to string yapılıyor.
    df['journey'] = [','.join(map(str, l)) for l in df[event]]
    journey=df['journey'].values
    journey=np.unique(journey,return_counts=True)
    df.drop(columns=['journey'],inplace=True)
    return df_with_journey
journey=find_journey(df) 
    
    