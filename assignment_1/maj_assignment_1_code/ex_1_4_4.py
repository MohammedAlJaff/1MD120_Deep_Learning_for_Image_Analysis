import numpy as np
import matplotlib.pyplot as plt

from maj_linear_model import LinearRegressionModel, standarize_data
from load_auto import load_auto


if __name__ =='__main__':
    # Load automobile data-set 
    Xraw, y = load_auto()
    
    
    horsepower_column_j = 2
    X1_raw = Xraw[:,horsepower_column_j].reshape([np.size(Xraw[:,horsepower_column_j]),1])
    X1_std = standarize_data(X1_raw)
    
    
    # UNSTANDARDIZED
    lr = 0.00001
    
    model_raw = LinearRegressionModel(data_X=X1_raw, true_label_Y=y)
    w1_raw, b1_raw, j1_raw =  model_raw.train_linear_model(X=X1_raw, y_true=y, 
                                                 nr_iter=4000,
                                                 learning_rate=lr,
                                                 verbose = True)
    
    
    
    
    lr = 0.00001
    
    model_std = LinearRegressionModel(data_X=X1_std, true_label_Y=y)
    w1_std, b1_std, j1_std =  model_std.train_linear_model(X=X1_std, y_true=y, 
                                                 nr_iter=4000,
                                                 learning_rate=lr,
                                                 verbose = True)
    
    
    
    plt.figure(figsize=[12,5])
    plt.plot(j1_raw, label='non-standardised features')
    plt.plot(j1_std, label='standardised features')
    
    plt.title(f'Gradient decent learning rate: {lr}')
    plt.xlabel('nr or iteration')
    plt.ylabel('Cost')
    plt.legend()

    

    y_pred_raw = model_raw.predict(X1_raw)
    y_pred_std = model_raw.predict(X1_std)
    
    
    
    plt.figure(figsize=[12,5])
    plt.scatter(X1_raw, y)
    
    plt.plot(X1_raw, y_pred_raw, 'kx', 
             label =  'data-point predictions')
    plt.plot(X1_raw, y_pred_raw, '-r', 
             label =  'line eq: '+str(w1_raw[0,0])[:6]+'x + ' + str(b1_raw)[:5])
    
    plt.legend()
    
    plt.title(f'Model 1: Horsepower v. MPG a & best-fit line from grad-decent optimization - lr = {lr}')
    plt.ylabel('miles per gallon (mpg) ')
    plt.xlabel('horsepower (Standardized values)')
    
    plt.savefig('ex_1_4_4_Model1_horsepower_vs_mpg_NOT_STANDARDISED.png')
    
    
    plt.show()
