import numpy as np
import matplotlib.pyplot as plt

from maj_linear_model import LinearRegressionModel, standarize_data
from load_auto import load_auto



if __name__ =='__main__':
    # Load automobile data-set 
    Xraw, y = load_auto()
    
    # Standardize data matrix
    X7 = standarize_data(Xraw)
    
    ### 
    learning_rates = [0.19, 1e-1, 1e-3, 1e-4, 1e-2]
    
    training_curves = []

    for i in range(len(learning_rates)):

        #define model 
        maj_7_model = LinearRegressionModel(data_X=X7, true_label_Y=y)

        # fit model with learning rate 
        lr = learning_rates[i]
        w7, b7, j7 =  maj_7_model.train_linear_model(X=X7,
                                                     y_true=y, 
                                                     nr_iter=1000, 
                                                     learning_rate=lr)
        training_curves.append(j7)
        # same and append trainng cost trajectories
        
    
    print('weights w: ', maj_7_model.W)
    print('offset b: ', maj_7_model.b)
    
    plt.figure(figsize=[15,5])
    for i in range(len(learning_rates)):
        plt.plot(training_curves[i], label='$\\alpha$ = '+str(learning_rates[i]))
        plt.legend()

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Training Cost/Emperical Risk')
    plt.title('Model 2: Using all 7 features as the input data')
    
    plt.show()