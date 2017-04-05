import os
os.chdir(r'C:\Users\Xie-Kaiqiang\Desktop\NTUEE ML 2016\all\all\HW\hw1')
from pre_process import create_feature_label
#from linear_regression import my_lr_train, my_lr_test


os.chdir(r'C:\Users\Xie-Kaiqiang\Desktop\NTUEE ML 2016\all\all\HW\hw1\data')
train_data = create_feature_label('train.csv')
test_data = create_feature_label('test_X.csv')

#weight,biases = my_lr_train(train_data,learning_rate=0.1,thelta=0.15,max_iter=1000,normalize=True)

#results =  