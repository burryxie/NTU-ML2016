import os
import chardet
import numpy as np
import pandas as pd
os.chdir(r'C:\Users\Xie-Kaiqiang\Desktop\NTUEE ML 2016\all\all\HW\hw1\data')


def create_feature_label(filename):
    
    feature_label = pd.DataFrame()
    # get the encoding of the file
    with open(filename,'rb') as f:
        enc_res = chardet.detect(f.read())

    train_data = pd.read_csv(filename,encoding = enc_res['encoding'])
    del enc_res


    labels = pd.Series.unique(train_data.ix[:,'indicator']) #fuck python

    for i in range(0,int(train_data.shape[0]/len(labels))): 
        #daily data
        daily_train = train_data.ix[(i*len(labels)):(i*len(labels)+17),:]
        daily_train.index = daily_train.ix[:,'indicator']
        date = pd.Series.unique(daily_train.ix[:,0])[0]
        pm_25 = daily_train.ix[9,2:]  #pm 2.5
        daily_train = daily_train.drop(['RAINFALL','PM2.5'])
        daily_train = daily_train.drop(['indicator','date'],1)
    
        if(filename == 'test_X.csv'):
            sample_data = daily_train
            sample_data = sample_data.as_matrix()
            sample_data = sample_data.flatten()
            target_pm25 = pm_25[0:9]
            target_pm25 = np.array(target_pm25)
            sample_data = np.insert(sample_data,0,target_pm25).reshape([1,153])
            sample_data = np.insert(sample_data,0,date).reshape([1,154])
            feature_label = feature_label.append(pd.DataFrame(sample_data),ignore_index=True)
            
            
        if(filename == 'train.csv'):
            for j in range(daily_train.shape[1]-9):
                # create features 
                sample_data =  daily_train.ix[:,j:(j+9)]
                sample_data = sample_data.as_matrix()
                sample_data = sample_data.flatten()
                
                target_pm25 = int(pm_25[j+9])
                target_pm25 = np.array(target_pm25)
                sample_data = np.insert(sample_data,0,pm_25[j:(j+9)])
                
                sample_data = np.insert(sample_data,0,target_pm25).reshape([1,154])
                sample_data = np.insert(sample_data,0,date).reshape([1,155])
                feature_label = feature_label.append(pd.DataFrame(sample_data),ignore_index=True)         
            

    if(filename == 'train.csv'):
        names = ['date','target_pm25']
        names = names+['feature'+'_'+str(i) for i in range(feature_label.shape[1]-2)]
        
    if(filename == 'test_X.csv'):
        names = ['date']
        names = names+['feature'+'_'+str(i) for i in range(feature_label.shape[1]-1)]   
            
    feature_label.columns = names    
    return(feature_label)


    




        


        
        
        
        
    