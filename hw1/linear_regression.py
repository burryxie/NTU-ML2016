import numpy as np
import pandas as pd

def my_lr_train(data,learning_rate=0.05,thelta=0.03,max_iter = 1000,normalize = True):
    num_coef = data.shape[1]-2
    num_expl = data.shape[0]
    train_x = data.ix[:,2:(data.shape[1])]
    train_y = data.ix[:,1]
    train_x = train_x.apply(pd.to_numeric,errors='ignore').as_matrix()
    train_y = pd.to_numeric(train_y).as_matrix()
    train_y = train_y.reshape([3600,1])
    
    #normalizetion
    if(normalize == True):
        row_mean = np.mean(train_x,axis=0).reshape([1,num_coef])
        row_std  = np.std(train_x,axis=0).reshape([1,num_coef])
        train_x  = (train_x - row_mean) / row_std
    
    train_x_t = train_x.transpose()
    weight = np.random.normal(size=[num_coef,1])
    biases = np.random.normal(size=[1,1])
    loss = 0

    y_hat = np.dot(train_x,weight)+biases
    loss =  1/num_expl * 0.5*sum(np.square(train_y - y_hat)) + thelta*sum(np.square(weight))

    #lurk =0
    
    for i in range(1,max_iter):
        print('interation {0}  --- loss value {1}'.format(i,loss[0]))      
        
        #calculate gradient
        grad_weight = np.dot(train_x_t,y_hat - train_y) + 2*thelta*weight
        grad_biases = sum(y_hat - train_y)
        
        weight = weight - learning_rate * 1/num_expl * grad_weight
        biases = biases - learning_rate * 1/num_expl * grad_biases
        
        y_hat = np.dot(train_x,weight) + biases
        new_loss = 1/num_expl * 0.5*sum(np.square(train_y - y_hat)) + thelta * sum(np.square(weight))
        
        #if(np.abs(new_loss-loss) < 1e-6):
         #   lurk += 1
        #if(lurk > 10):
         #   print('lurk is above 10!')
          #  break 
       # if(new_loss > loss):
        #    print('loss is getting bigger,new loss is {0}, old loss is {1}'.format(new_loss,loss))
         #   break
        loss = new_loss       
                     
    return(weight,biases)

w,b = my_lr_train(train_data)
#
#
#
#def my_lr_test(weight,biases,data,normalize=True):
#    my_id = data.ix[:,0]
#    data = data.drop(['date'],aixs=1)
#    
#    data = data.apply(pd.to_numeric,errors='ignore').as_matrix()
#    #num_test = data.shape[0]
#    num_coef = data.shape[1]
#    
#    if(normalize == True):
#        row_mean = np.mean(data,axis=0).reshape([1,num_coef])
#        row_std  = np.std(data,axis=0).reshape([1,num_coef])
#        data  = (data - row_mean) / row_std
#        
#    res = np.dot(data,weight)+biases
#    
#    my_res = pd.DataFrame(my_id)
#        


        
        
        
    
    

