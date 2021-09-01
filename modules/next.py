import pandas as pd
import dataset as d_module
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

df=d_module.df
unique_activity=d_module.unique_dict[1]
id=d_module.df_columns_name[0]
columns_name=list(df.columns)
event=columns_name[1]

shifted = (
    df.groupby(id)
      .agg({event: lambda s: list(zip(s.shift(1), s))[1:]})
      #.apply(lambda lis: any([els == ("activity_05", "activity_03") for els in lis]), axis=1)
      .explode(event)[event].value_counts()
)
shifted.sort_index(inplace=True)

prob_dict=dict()
sum=0
for j in unique_activity:
    for i in range(len(shifted)):
        if j==shifted.index[i][0]:
            sum+=shifted[i]
    prob_dict[j]=sum
    sum=0
# Prob Matrix
l=list()
for key,value in prob_dict.items():
    for j in range(len(shifted)):
        if key == shifted.index[j][0]:
            l.append(shifted[j]/value)
prob_df=pd.DataFrame([(0,l[0],l[1],0,0,0),(0,0,l[2],l[3],0,0),(l[4],0,0,l[5],l[6],l[7]),\
                      (l[8],0,0,0,0,0),(0,l[9],0,0,0,0),(0,0,0,0,0,0)],index=unique_activity,columns=unique_activity)



# bir aktiviteden sonra gelen diğer aktiviteler
df.drop(columns=[columns_name[0],columns_name[1]],inplace=True)
def eventler(num):
    if num==1:
        activity=d_module.model_encoding(df,'activity_02','activity_03','80yo+','middle','no')
        RandomForest_binary(activity)
        return activity
    elif num==2:
        activity=d_module.model_encoding(df,'activity_03','activity_04','60-79yo','high','no')
        RandomForest_binary(activity)
        return activity
    elif num==3:
        activity=d_module.model_multi_encoding(df,'activity_01','activity_04','activity_05','activity_07',\
                                         '80yo+', '0-19yo', 'middle', 'low', 'no', 'yes')
        RandomForest_multi(activity)
        return activity
    elif num==4:
        activity=d_module.model_multi_encoding(df,'activity_01','activity_01','activity_04','activity_01','40-59yo','60-79yo',\
                                         'middle','high','yes','no')
        RandomForest_binary(activity)
        return activity
    else:
        activity=d_module.model_encoding(df,'activity_05','activity_02','60-79yo','high','no')
        RandomForest_binary(activity)
        return activity

# -------------------------MODEL-------------------------------
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
mtf = MultiOutputClassifier(rfc, n_jobs=-1)

def RandomForest_binary(df):
    X=df.iloc[:,0:8]
    Y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
    
    rfc.fit(x_train,y_train)
    y_pred=rfc.predict(x_test)
    
    print("Model - Random Forest(entropy) Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
# çoklu sınıflandırma
def RandomForest_multi(df):
    X=df.iloc[:,0:8]
    Y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
    
    mtf.fit(x_train, y_train.values.reshape(-1,1))
    y_pred=mtf.predict(x_test)
    print("Model multi - Random Forest Accuracy:{0}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
a=list()    
output = list()
journey_list=list()
# yeni gideceği aktiviteyi hesaplar
def find_class(df,customer,step):
    no_joruney=False
    path=0
    for i in df:
        if np.array_equal(customer, i[0:8]):
            path=i[-1]
            break
        else:
            path=0
    step-=1
    for x in journey_list:
        if x not in output:
            output.append(x)
    if step==0 or no_joruney:
        print( 'Yolculuk bitti')
    if len(output) ==1 and len(journey_list)>3:
        a.append(output[0])
    if len(output)==2  and len(journey_list)>3:
        no_joruney=True
    else:
        journey_list.append(path)
        return forecast(customer,step,path)

# yeni gelen müşteri için x adım sonrası
def forecast(customer,step,path):
    customer=np.array(customer)
    new_path=path
    if not journey_list:
        y_pred=rfc.predict([customer])
        journey_list.append(y_pred[0])
        activity=eventler(y_pred)
        activity=activity.values
        new_path=find_class(activity, customer,step)
    if step <0:
        print ('Hata')
    if step >=1:
        activity=eventler(new_path)
        activity=activity.values
        new_path=find_class(activity, customer,step)
                
activity_1=eventler(1)
activity_2=eventler(2)
activity_3=eventler(3)
activity_4=eventler(4)
activity_5=eventler(5)
step_size=10
forecast([0,0,0,1,0,1,0,0],step_size,0)
journey_list=journey_list[0:step_size]    