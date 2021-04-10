import numpy as np
import matplotlib.pyplot as plt

from maj_linear_model import LinearRegressionModel, standarize_data
from load_auto import load_auto


if __name__ =='__main__':
    # Load automobile data-set 
    Xraw, y = load_auto()
    
    # Standardize data matrix
    X = standarize_data(Xraw)

    horsepower_column_j = 2
    X1 = X[:,horsepower_column_j].reshape([np.size(X[:,horsepower_column_j]),1])

    
    
    ### 
    learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    training_curves = []

    for i in range(len(learning_rates)):

        #define model 
        maj_1_model = LinearRegressionModel(data_X=X1, true_label_Y=y)

        # fit model with learning rate 
        lr = learning_rates[i]
        w1, b1, j1 =  maj_1_model.train_linear_model(X=X1, y_true=y, nr_iter=1000, learning_rate=lr)
        training_curves.append(j1)
        # same and append trainng cost trajectories


    plt.figure(figsize=[15,5])
    for i in range(len(learning_rates)):
        plt.plot(training_curves[i], label='$\\alpha$ = '+str(learning_rates[i]))
        plt.legend()

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Training Cost/Emperical Risk')
    plt.title('Model 1: Using only  "Horsepower" as the input data feature')

    
    lr = 0.01
    maj_1_model = LinearRegressionModel(data_X=X1, true_label_Y=y)
    w1, b1, j1 =  maj_1_model.train_linear_model(X=X1, y_true=y, nr_iter=1000, learning_rate=lr)
    
    y_pred = maj_1_model.predict(X1)
    
    plt.figure(figsize=[12,5])
    plt.scatter(X1, y)
    
    plt.plot(X1, y_pred, 'kx', label =  'data-point predictions')
    plt.plot(X1, y_pred, '-r', label =  'line eq: '+str(w1[0,0])[:6]+'x + ' + str(b1)[:5])
    
    plt.legend()
    
    plt.title(f'Model 1: Horsepower v. MPG a & best-fit line from grad-decent optimization - lr = {lr}')
    plt.ylabel('miles per gallon (mpg) ')
    plt.xlabel('horsepower (Standardized values)')
    plt.savefig('ex_1_4_Model1_horsepower_vs_mpg.png')
    
    
    plt.show()
    
