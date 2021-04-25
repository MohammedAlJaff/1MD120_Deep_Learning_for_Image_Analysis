# AUTHOR: Mohammed Al-Jaff

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def maj_load_mnist():
    '''
    Loads the MNIST dataset fro both test and train sets from the cvs version of the dataset.
    This version of MNIST can be downloaded from: 
    
    https://pjreddie.com/projects/mnist-in-csv/
    
    return: train_images, train_labels, test_images, test_labels
    '''
    # Get MNIST train and test intonumpy arrry form which is already in flat-form. 
    
    train_255_dataset = pd.read_csv('mnist_train.csv', 
                                    delimiter=',',
                                    header=None).to_numpy()
    
    test_255_dataset = pd.read_csv('mnist_test.csv',
                                   delimiter=',',
                                   header=None).to_numpy()
    
    # Convert from [0,255] pixel value format to [0,1] format
    train_images = train_255_dataset[:, 1:]/255.0
    test_images = test_255_dataset[:, 1:]/255.0
    
    # Extact 1 column where labels are.
    train_labels = train_255_dataset[:,0]
    test_labels = test_255_dataset[:,0]
    
    train_one_hot_labels = np.zeros([train_labels.size, 10])
    for i in range(train_labels.size):
        train_one_hot_labels[i,int(train_labels[i])] = 1

    
    test_one_hot_labels = np.zeros([test_labels.size, 10])
    for i in range(test_labels.size):
        test_one_hot_labels[i,int(test_labels[i])] = 1

    
    return train_images, train_one_hot_labels, test_images, test_one_hot_labels
    
    
def generate_toy_data(N_third=100, x_dim=4):
    
    c1 = np.random.normal(0,1, size=(N_third,x_dim))
    yc1 = np.tile([1,0,0], [N_third,1])

    c2 = np.random.normal(0,3, size=(N_third,x_dim))
    yc2 = np.tile([0,1,0], [N_third,1])

    c3 = np.random.normal(0,2, size=(N_third,x_dim))
    yc3 = np.tile([0,0,1], [N_third,1])

    
    c1[:,0] += 5 
    c1[:,1] += 13
    c1[:,2] += 13

    c2[:,0] += -5 
    c2[:,1] += -10
    c3[:,2] += -10

    c3[:,0] += -5 
    c3[:,1] += 10
    c3[:,2] += -13
    
    N = 3*N_third

    X = np.vstack([c1, c2, c3])
    y = np.vstack([yc1, yc2, yc3])
    
    
    rand_indx_perm = np.random.permutation(N)
    #print(rand_indx_perm)
    
    
    y = y[rand_indx_perm]
    X = X[rand_indx_perm]
    
    return X, y