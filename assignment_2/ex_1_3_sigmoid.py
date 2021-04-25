from maj_load_mnist import maj_load_mnist
import numpy as np 
import matplotlib.pyplot as plt
import majnn 


if __name__=='__main__':
    print('Run/Script: ex_1_3_sigmoid')
    
    print('Loading train and test dataset to memory\n')
    train_imgs, train_lbls, test_imgs, test_lbls = maj_load_mnist()
    
    hidden_architecture = [100]
    
    
    mnn, jtrn, jtst, acctrn, acctst = majnn.create_and_train_NN(train_imgs, train_lbls,
                                                                test_imgs, test_lbls,
                                                                hidden_architecture,
                                                                activation_function= 'sigmoid',
                                                                learning_rate = 0.4, 
                                                                minibatch_size = 30,
                                                                nr_epochs=3, 
                                                                verbose=True)
    
    majnn.visualize_training(mnn, jtrn, jtst, acctrn, acctst)
    # Visualization: Plots for training phase (Cost and accuracy over train and test data)
   
   