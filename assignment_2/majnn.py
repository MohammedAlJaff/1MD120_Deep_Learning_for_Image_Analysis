from maj_load_mnist import maj_load_mnist
import numpy as np 
import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng()


# Status: Done
def accuracy(y_true_class, y_pred_class):
    '''
    Classification Accuracy calculation between to one-hot-encoded label vectors 
    input: 
        - y_true_class: array
        - y_pred_class: array 
    '''
    dummy_a = 0 #True positive counter
    for i in range(y_true_class.size):    
        if y_true_class[i] == y_pred_class[i]:
            dummy_a += 1 
    
    return dummy_a/y_true_class.size


# Status: Done
def sigmoid(x):
    '''
    input:
        - x (ndarry):   
    output:
        Sigmmoid values. Element wise for each entry in x.
    '''    
    exp_x = np.exp(x)
    return exp_x/(1+exp_x)


# Status: Done
def derivative_sigmoid(x):
    '''
    Derivate of above sigmoid function
    '''
    return sigmoid(x)*(1-sigmoid(x))

# Status: Done
def relu(x):
    '''
    Implements the ReLU functions which returns the max of [0, x] elementwise
    input:
        - x: array
    output: 
        elementwise relu on x
    '''    
    return np.maximum(x, 0)

# Status: Done
def derivative_relu(x):
    '''
    Implements the derivative of the ReLU function above.
    '''    
    return np.heaviside(x, 0)



# Status: Done
class MAJNN(): 
    # status: 
    def __init__(self, hidden_architecture, X, y_output_dim, usr_activation_function):
        '''
        Input: 
            - Architecture input (list): - Length of list is the number of hidden layers 
                and the value of each element corresponds to the  nr of nodes in that layers
            - X input data (array) Determines the input size of network
            - y 'labels'(array): determines the output layer size of network.
            - activation_function:  either 'relu' or 'sigmoid'. Raises error for any other string.
        '''
        
        # Construct newtwork weights and biases matrices from X,y
        # and hidden_architecture 
        self.__initialize_parameters(X, hidden_architecture, y_output_dim)
        
        # Assign and determine activation function
        self.__set_activation_function(usr_activation_function)
        
        
    # Status: 
    def __initialize_parameters(self,X, hidden_architecture, y_output_dim):
        '''
        Initializes all network parameters: A weight's matrix and 
        a bias vector for each layer of the network. 
        
        The shape of a layers weight matrix and bias vector 
        '''
        
        # Vector representation of all layers of network
        # first element is the number of input data features
        # last element is the output dimension
        # all elements in between corrspond to the number of hidden nodes insiide each layer.s
        all_layers_arch = [ X.shape[1], *hidden_architecture, y_output_dim]
        self.network_architecture = all_layers_arch
        
        print(f'whole network architecture (nr of nodes in each layer): {all_layers_arch}\n')
        
        # Extract Nr layers in the convential sense, ie without input layer.
        self.nr_of_layers = len(all_layers_arch)-1 
        
        bias_vectors = [] # storing the bias vectors for each layer.
        weight_matricies = [] # storing the weights matrix for each layers
        outputs_h = [] # used only for debugging and sense making. 
        
        # Consrtuction of weight matrix for each layer based on given architechture
        for l in range(1, len(all_layers_arch)):
            print(f"\tlayer: {l}")
            # Weight matrix init for layer l:
            # Col and Row number determination
            w_nr_rows = all_layers_arch[l]  # nr of columns equals nr or nodes in current layer
            w_nr_col = all_layers_arch[l-1] # nr of columns equals nr of 
            w = rng.normal(loc=0.0, scale=0.01, size=[w_nr_rows, w_nr_col]) # random inital weights.

            # Bias vector init for layer l:
            b = np.zeros(all_layers_arch[l])
            b = b.reshape((b.size, 1))

            a = np.matmul(X, w.T)
            X = a

            print(f"\t-Bias vector size of layer {l}: {b.shape}")
            print(f"\t-Weight matrix shape of layer {l}: {w.shape}")
            print(f'\t-Shape of outpute of layer {l}: {a.shape}')
            print('')

            bias_vectors.append(b)
            weight_matricies.append(w)
            
            #self.all_b[l]
            #self.all_W[l]
            
        self.network_layer_weights_W = weight_matricies
        self.network_layer_biases_b = bias_vectors
        
    # Status:
    def __set_activation_function(self, usr_option):
        ''' 
        Internal function to fix 
        '''
        available_activation_functions = [sigmoid, relu]
        
        if usr_option == 'sigmoid':
            self.activation_function = sigmoid
            self.derivative_activation_function = derivative_sigmoid
            
            self.activation_function_type = 'sigmoid'
            print('should print out 0.8807970779778824: ', self.activation_function(2))
            print('Activation function: Sigmoid')
            
        elif usr_option == 'relu':
            self.activation_function = relu
            self.derivative_activation_function = derivative_relu
            
            self.activation_function_type = 'relu'
            
            
            print('should print out 2: ', self.activation_function(2))
            print('Activation function: Relu')
        else: 
            raise ValueError('Men hallå! specify correct activation_function in class arguments')
            
    # Status:        
    def nr_params(self):
        '''
        Return: 
            - Nr of parameters for specifed newtork (integer)
        '''
        nr_params = 0
        for el in self.network_layer_biases_b:
            nr_params += el.size 
    
        for el in self.network_layer_weights_W: 
            nr_params += el.size

        return nr_params
    
    # Status:
    def __linear_forward(self, X, W, b, verbose=False):
        # ensure compatible dim
        # make sure 
        if verbose:
            print(f'\t\tshape of input_X: {X.shape}' )
            print(f'\t\tshape of W.T: {W.T.shape}' )
            print(f'\t\tshape of b: {b.T.shape}\n' )
        
        # row wise row-vector sutraction
        a = np.matmul(X, W.T) + b.T
        return a
    
    # Status:
    def __activation_forward(self, a):
        return self.activation_function(a)
    
    # Status:  
    # status: Implement Raw last layer output before softmax
    def model_forward(self, X, verbose=False):
        self.linear_outputs = []
        self.activated_outputs = []
        
        dummy_l = 0
        for layer_nr in range(len(self.network_layer_weights_W)-1):
            #print(f'\ncomputing layer nr: {layer_nr}')
            W = self.network_layer_weights_W[layer_nr]
            b = self.network_layer_biases_b[layer_nr]
            
            a = self.__linear_forward(X, W, b)    
            #print(f'shape of a: {a.shape}')    
            self.linear_outputs.append(a)
            
            o = self.__activation_forward(a)    
            #print(f'shape of o: {o.shape}')
            self.activated_outputs.append(o)
            
            X = o
            dummy_l += 1
            if verbose:
                print(f'\ncomputing layer nr: {layer_nr}')
                print(f'shape of a: {a.shape}')
                print(f'shape of o: {o.shape}')
                     
        # last layer output 
        W = self.network_layer_weights_W[dummy_l]
        b = self.network_layer_biases_b[dummy_l]
        a = self.__linear_forward(X, W, b)    

        self.linear_outputs.append(a)

        # We do not apply non-linearities to last layer output
        # the final layer output is simply the dot-produc-sum calc 
        # without activation function applied.
        o = a    
        self.activated_outputs.append(o)
        
        if verbose:
            print(f'\ncomputing last layer: {dummy_l}')
            print(f'shape of a: {a.shape}')
            print(f'shape of o: {o.shape}')
        
        return o    
    
    
    #Status: ACCOUNT FOR HUGE VALUES OF X
    def softmax(self, z):
        ## magnetidue correction for huge z!!!
        ## Needed for numerical stability.
    
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z, axis=1)
        sm_z = exp_z/sum_exp_z.reshape((sum_exp_z.size,1))
        return sm_z
        
    # Status:
    def predict(self, X):
        '''
        Compute and return predictions with current network weights. 
        '''
        o_last_layer = self.model_forward(X)
        return self.softmax(o_last_layer)
    
    
    # Status: Implement clipping of preds
    def compute_loss(self, y_true, y_pred, epsilon=0.00001):
        '''
        Computes the cross entroupy loss between true one-hot encoded label vectors and preditictions
        input: 
            - y_true, array
            - y_pred: array
            - epsilon: positive real scaler. Used to ensure that we remove 
            any instance of 0 in our predictions and thus avoid -inf.
        # Anecdote: I once had a internship where this clipping issue ruined a whole good spring weekend. 
        '''
        
        y_pred = np.clip(y_pred, a_min=epsilon, a_max= 1-epsilon) #clip predictions incae 
        total_cross_entropy_loss = -1*np.sum(y_true*np.log(y_pred)) #compute total cross entropy loss 
        avg_loss = total_cross_entropy_loss/(y_true.shape[0]) # average loss over dataset.
        return avg_loss

    
    # Status: Done
    def update_one_step(self, X, y_true, alpha = 0.1, cost_verbose = False, verbose=False, return_grads = False):
        '''
        Performs one parameter-update using batch gradient decent over provied data X and y.
        input: 
            - X, data matrix with all x's
            - y_true, - true labels (one hot encoded)
            - alpha = 0.1. Learning rate update. 
            - cost_verbose = False. If True, dispays cost of each mini-batch update step. 
            - verbose=False, - if True, displays detailed information about the update step. 
                                used fo debugging.
            - return_grads = False - If true  Used for debugging and sense-making when running experiments. 
        
        output: 
            - None: Changes state of NN object in terms of update of weights. 
            or..
            gradient values if return_grads is set to true. 
     
        '''
        N = X.shape[0]
        # Predict
        y_hat = self.predict(X)
        
        if cost_verbose:
            #compute cost (jsut for show)
            print(f"cost: {self.compute_loss(y_pred=y_hat, y_true=y_true)}")

        # compte last layers error vector
        delta_last_layer = y_hat - y_true

        if verbose:
            print('Nr of layers in network: ', self.nr_of_layers, '\n')
        
        grads_W = []
        grads_b = []
        
        
        # For each layer . 
        # - Compute partial derivatie of each weight in layer 
        # - update layer weights 
        # - compute and make ready next layers error vector
        delta_last = delta_last_layer
        for k in reversed(range(self.nr_of_layers)):
            
            # Make sure we that we chose input data if we reach first hidden layer. 
            if verbose:
                print('\nk =', k)
                # update k'th weightmatrix
                print(f'shape of delta_last: {delta_last.shape}')
            
            if k < 1:
                #pick input for o_k-1
                o_k_minus_1 = X
            else:
                o_k_minus_1 = self.activated_outputs[k-1]     
            
            if verbose:
                print(f'shape of o_k_minus_1: {o_k_minus_1.shape}')
            
            dJdW_k = np.matmul(delta_last.T, o_k_minus_1) / (N)
            
            dJdb_k = np.matmul(delta_last.T, np.ones([o_k_minus_1.shape[0],1])) / (N)
            
            grads_W.append(dJdW_k)
            grads_b.append(dJdb_k)
            
            if verbose:
                print(f'shape of dJdW_k: {dJdW_k.shape}')
                print(f'shape of dJdb_k: {dJdb_k.shape}')
            
            # get current layers parameters
            W_k = self.network_layer_weights_W[k]
            b_k = self.network_layer_biases_b[k]
            
            if verbose:
                print(f'shape of W_k: {W_k.shape}')
                print(f'shape of b_k: {b_k.shape}')
            
            # Compute updates weight / params
            W_k_new = W_k - alpha*dJdW_k
            b_k_new = b_k - alpha*dJdb_k
            
            if verbose:
                print('')
                print(W_k_new-W_k)
            
            # update delta for next iteration
            if k < 1:
                #pick input for o_k-1
                a_k_minus_1 = X
                #g_prime_a_k_minus_1 = sigmoid(a_k_minus_1, derivative=True)
                g_prime_a_k_minus_1 = self.derivative_activation_function(a_k_minus_1)            
            else:
                a_k_minus_1 = self.linear_outputs[k-1]
                #g_prime_a_k_minus_1 = sigmoid(a_k_minus_1, derivative=True)
                g_prime_a_k_minus_1 = self.derivative_activation_function(a_k_minus_1)
                
            if verbose: 
                print(f"\tshape of a_k_minus_1:  {a_k_minus_1.shape}")    
                print(f"\tshape of gs'(a_k_minus):  {g_prime_a_k_minus_1.shape}")
            
            # Compute next back-ward error/delta:
            delta_last = g_prime_a_k_minus_1 * np.matmul(delta_last, W_k)
            
            # Replace old weights with new updated weights. 
            self.network_layer_weights_W[k] = W_k_new
            self.network_layer_biases_b[k] = b_k_new
            
        if return_grads:
            return grads_W, grads_b
        
    # Status: Done     
    def train_model(self, X_train, y_train, X_test, y_test, 
                   learning_rate = 0.4, minibatch_size = 30, nr_epochs=3, verbose=True):
        
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        nr_minibatches = int(np.floor(N_train/minibatch_size))
        
        nr_of_test_points = int(0.1*N_test) # used to select random subset of test dataset to eval performance
        
        
        if verbose:
            print('- Nr of data-points (rows): ', N_train)
            print('- Nr of minibatches per epoch: ', nr_minibatches)
            print(f"- Minibatch size: {minibatch_size}")
            
        # 
        J_train = np.zeros(nr_epochs*nr_minibatches)
        J_test = []
        
        acc_train = []
        acc_test = []
        
        dummy_J_index = 0
        
        for epoch_i in range(nr_epochs):
            
            print(f'Epoch - {epoch_i+1}')
            
            epoch_permutation_train = np.random.permutation(list(range(N_train)))
            
            for minibatch_i in range(nr_minibatches):
                minibatch_indx = epoch_permutation_train[minibatch_size*minibatch_i:minibatch_size*(minibatch_i+1)]
                
                minibatch_X_train = X_train[minibatch_indx, :] # Correct variable names
                minibatch_y_train = y_train[minibatch_indx, :] # Correct variable names
                
                #minibatch gradient decent learning
                grads_W, grads_b = self.update_one_step(X=minibatch_X_train, 
                                               y_true = minibatch_y_train, 
                                               alpha = learning_rate, 
                                               return_grads = True)
                
                
                minibatch_y_preds = self.predict(minibatch_X_train)
                J_minibatch_i = self.compute_loss(y_true = minibatch_y_train,
                                                  y_pred = minibatch_y_preds)
                
                J_train[dummy_J_index] = J_minibatch_i
                
                
                # Every 100th update: evaluate network performance on subset of test-data
                if (minibatch_i%200==0):
                    
                    # first 
                    permutation_test_indx = np.random.choice(list(range(N_test)), nr_of_test_points)
                    
                    X_test_subset = X_test[permutation_test_indx, :]
                    y_test_subset = y_test[permutation_test_indx, :]
                    
                    
                    # Calc cost on test subset
                    y_pred_test = self.predict(X_test_subset)
                    J_test_i = self.compute_loss(y_true = y_test_subset,
                                                 y_pred = y_pred_test)    
                    J_test.append([dummy_J_index, J_test_i])
                    print(f'\tminibatch {minibatch_i+1}: \t cost: {J_minibatch_i}')
                    
                    
                    # Calc train and test accuracies:
                    minibatch_y_preds_labels = np.argmax(minibatch_y_preds, axis=1)
                    minibatch_y_true_labels = np.argmax(minibatch_y_train, axis=1)
                    
                    y_preds_labels_test = np.argmax(y_pred_test, axis=1)
                    y_true_labels_test = np.argmax(y_test_subset, axis=1)
                    
                    train_acc_i = accuracy(y_true_class = minibatch_y_preds_labels,
                                           y_pred_class = minibatch_y_true_labels)
                    
                    test_acc_i = accuracy(y_true_class = y_true_labels_test, 
                                          y_pred_class = y_preds_labels_test)
                    
                    
                    print(f'\t\tTrain Accuracy: {train_acc_i}')
                    print(f'\t\tTest Accuracy:  {test_acc_i}\n')
                    
                    acc_train.append([dummy_J_index, train_acc_i])
                    acc_test.append([dummy_J_index, test_acc_i])
                                    
                    if verbose: 
                        for g in grads_W: 
                            print('______ max(abs(x)) = ', np.linalg.norm(g, ord= np.inf))
                            
                            
                if verbose: 
                    print(f'\tminibatch {minibatch_i+1}: \t cost: {J_minibatch_i}')
                
                
                # CHECK ON THIS Later!
                dummy_J_index += 1 
                
                
        J_test = np.array(J_test)
        acc_train = np.array(acc_train) # conversion from standard 2d np array
        acc_test = np.array(acc_test) # conversion from standard 2d np array

        return J_train, J_test, acc_train, acc_test, 


# Status: Done
def create_and_train_NN(X_train, y_train, X_test, y_test,
                        hidden_architecture, activation_function, 
                        learning_rate = 0.4, 
                        minibatch_size = 30, nr_epochs=3, 
                        verbose=True): 
    '''
    Convenience function that contructs, trains and returns back a MAJNN neural network object.
    Input:
            - X_train, y_train, X_test, y_test,
            - hidden_architecture, 
            - activation_function: string : 
            - learning_rate : scaler
            - minibatch_size = 30, 
            - nr_epochs=3,
            - verbose=True):
    Output:
            - mnn: trained MAJNN neural network object.
            - jtrn, jtst: cost vs update-iterations vectors for test and train set
            - acctrn, acctst: accuracy vs. update iterations for test and train sets.
    '''
    
    #Extract Key constants
    train_N = X_train.shape[0] #nr of date points
    train_D = X_train.shape[1] # nr of input features
    train_y_dim = y_train.shape[1] # nr of output "features"
    
    print('N: ', train_N)
    print('D: ', train_D)
    print('number of classes: ', train_y_dim)
    
    # Instansiate MAJ neural network object
    mnn = MAJNN(X=X_train,
                hidden_architecture = hidden_architecture, 
                y_output_dim = train_y_dim, 
                usr_activation_function = activation_function)
    
    # Nr of params in network
    print(f"\nNr of params(weights and offsets): {mnn.nr_params()}")
    print(f"\nNr of layers: {mnn.nr_of_layers}")
    print(f"\nNr of weight matrices: {len(mnn.network_layer_weights_W)}")
    
    # Train network 
    jtrn, jtst, acctrn, acctst  = mnn.train_model(X_train, y_train,
                                                  X_test, y_test,
                                                  nr_epochs = nr_epochs,
                                                  learning_rate = 0.1,
                                                  verbose=False)
    
    return mnn, jtrn, jtst, acctrn, acctst


# Status: Done
def visualize_training(mnn, jtrn, jtst, acctrn, acctst):
    '''
    Convenience function to plot training/learning phase. 
    Plots two figures: 
        Figure 1: Cost value over update iterations.
        Figure 2: Accuracy over parameter update iterations 
        
    Both plots contain values for both the train and test datasets.
    '''
    
    # Plot Cost over training phase with test subset cost every K'th minibatch iteration.
    plt.figure(figsize=(12,8))
    plt.plot(jtrn, label='minibatch')
    plt.plot(jtst[:, 0], jtst[:, 1], label='test data subset', color='orange', linewidth=4)

    plt.title(f'Network architecture:   input-{mnn.network_architecture}-output with {mnn.activation_function_type} activation')
    plt.legend()
    plt.ylabel('Cost J / Emperical risk')
    plt.xlabel('minibatch gradient decent iteration')
    plt.grid()
    plt.show()
    
    # Plot Accuracy over training phase with test subset accuracy every K'th minibatch iteration.
    plt.figure(figsize=[12,6])
    plt.plot(acctrn[:, 0], acctrn[:, 1] , lineWidth=4, label='Train (minibatch) subset accuracy')
    plt.plot(acctst[:, 0], acctst[:, 1] , lineWidth=4, label='Test subset accuracy')
    plt.grid()
    plt.xlabel('Minibatch gradient decent updates (Iterations)')
    plt.ylabel('Accuracy (‰)')
    plt.show()
