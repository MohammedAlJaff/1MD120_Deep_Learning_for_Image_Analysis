from maj_load_mnist import maj_load_mnist
import numpy as np 
import matplotlib.pyplot as plt
import majnn 


if __name__=='__main__':
    print('Run/Script: ex_1_4_sigmoid')

    print('Loading train and test dataset to memory\n')
    train_imgs, train_lbls, test_imgs, test_lbls = maj_load_mnist()
    
    hidden_architecture = [400, 100, 25]
    
    mnn, jtrn, jtst, acctrn, acctst = majnn.create_and_train_NN(train_imgs, train_lbls,
                                                                test_imgs, test_lbls,
                                                                hidden_architecture,
                                                                activation_function= 'sigmoid',
                                                                learning_rate = 0.15, 
                                                                minibatch_size = 30,
                                                                nr_epochs = 20, 
                                                                verbose=True)
    
    # Visualization: Plots for training phase (Cost and accuracy over train and test data)
    majnn.visualize_training(mnn, jtrn, jtst, acctrn, acctst)
 
   