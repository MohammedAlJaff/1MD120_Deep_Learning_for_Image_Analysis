import numpy as np

class LinearRegressionModel():
    
    # status:
    def __init__(self, data_X, true_label_Y, learning_rate=0.01):
        
        self.__initialize_parameters(data_X)
        
        # gradient decent step-size
        self.learning_rate = learning_rate
        #print(f"nr of params: {self.nr_params}")
        
        
    # status:    
    def __initialize_parameters(self, X):
    
        self.nr_params = self.__determine_nr_model_params(X)
        
        self.b = 0
        self.b = 0.01*np.random.normal(1)
        
        self.W = np.zeros([self.nr_params-1, 1]) 
        self.W = 0.01*np.random.normal(size=self.nr_params-1).reshape([self.nr_params-1, 1])
    
    
    # status:
    def __determine_nr_model_params(self, X):
        return X.shape[1] + 1 # Number of features + extra for ofs
    
    
    # status:
    def nr_of_params(self):
        return self.nr_params
    
    
    # status:        
    def model_forward(self, X):
        
        if self.W.shape[0] != X.shape[1]:            
            raise Exception('Data matrix is wrong shape')
        
        preds = np.matmul(X, self.W) + self.b
        
        preds = preds.reshape([np.size(preds),1])
                              
        #print(f"[model forwards]- preds shape: {preds.shape}")
        
        return preds
    
    def predict(self, X):
        return self.model_forward(X)
    
    
    # status: 
    def compute_cost(self, X, y_true):
        
        # Dataset size
        N = X.shape[0]
        
        #print(f"compute_cost N: {N}")
       
        y_preds = self.model_forward(X)
        #print(f"compute_cost sum of y_preds: {np.sum(y_preds)}")
        #print(f"compute_cost of y_preds shape: {y_preds.shape}")
        #print(f"compute_cost sum of y_true: {np.sum(y_true)}")
    
        y_true = y_true.reshape([np.size(y_true),1])
        reg_err_vec = (y_true-y_preds)
        #print(f"compute_cost reg_erro_vec shape: {reg_err_vec.shape}")
        
        cc_cost = np.sum(np.square(y_true-y_preds)) / N
        #print(f"compute_cost cost: {cc_cost}")
        
        return cc_cost
    
    
    
    def J(self, X, y_true, w, b):
        
        N = X.shape[0]
        #print(f"__J N: {N}")
        y_preds = np.matmul(X, w) + b
        y_preds = y_preds.reshape([np.size(y_preds),1])
        
        y_true = y_true.reshape([np.size(y_true),1])
        
        #print(f"J sum of y_preds: {np.sum(y_preds)}")
        #print(f"J sum of y_preds shape: {y_preds.shape}")
        
        #print(f"J sum of y_true: {np.sum(y_true)}")
        
        reg_err_vec = (y_true-y_preds)
        #print(f"J reg_erro_vec shape: {reg_err_vec.shape}")
        
        J_cost = np.sum(np.square(y_true-y_preds)) / N
        #print(f"J cost: {J_cost}")
        return J_cost
    
    
    
    
    def grad_J(self, X, y_true):
        
        N = X.shape[0]
        #print(f"N: {N}")
        
        y_preds = self.model_forward(X)
        
        #print(f'[grad_J function]shape of y_preds: {y_preds.shape}')
        
        s = 2*(np.matmul(X.T, (y_preds-y_true)))
        #print(f"s shape: {s.shape}")
    
        grad_J_w = s/N
        grad_J_b = (2*np.sum(y_preds-y_true))/N
        
        
        return grad_J_w, grad_J_b
    
    
    def update_parameters(self, X, y_true, learning_rate=0.01):
        
        grad_J_w, grad_J_b = self.grad_J(X, y_true)    
        self.W = self.W  - learning_rate*grad_J_w
        self.b = self.b  - learning_rate*grad_J_b
    
    def train_linear_model(self, X, y_true, nr_iter, learning_rate=0.01):
        
        training_J = np.zeros(nr_iter)
        
        for i in range(nr_iter):
            self.update_parameters(X, y_true, learning_rate)
            training_J[i] = self.compute_cost(X, y_true)
            
            
        print('done')
        
        return self.W, self.b, training_J
    
    
    def numerical_grad_J(self, X, y_true, h=0.0001):
        
        W =  self.W
        b = self.b
        
        num_grad_J_w = []
        for j in range(np.size(W)):
            
            dW = np.zeros_like(W)
            dW[j] = h
            
            W_plus_dW = W+dW
        
            num_grad_J_w_j = (self.J(X, y_true, W_plus_dW, b) - self.J(X, y_true, W, b))/h        
            num_grad_J_w.append(num_grad_J_w_j)

            #print(num_grad_J_w_j)
        
        
        num_grad_J_w  = np.array(num_grad_J_w).reshape(W.shape)
        
        db = h
        num_grad_J_b = (self.J(X, y_true, W, b+db) - self.J(X, y_true, W, b)) / h
        #print(num_grad_J_b)
        
        return num_grad_J_w, num_grad_J_b 
            
    
    
        
    
    def update_params_one_step_numeric(self, X, y_true):
        
        grad_J_w, grad_J_b = self.numerical_grad_J(X, y_true)
        
        self.W = self.W  - self.learning_rate*grad_J_w
        self.b = self.b  - self.learning_rate*grad_J_b
        
    
    
    
    def replace_params(self, W_new, b_new):
        if W_new.shape != self.W.shape:
            msg = f"Model W shape: {self.W.shape}. Replacement shape {W_new.shape}"
            raise Exception(f"Error in new param dimensions\n {msg}")
        
        self.W = W_new
        self.b = b_new
        
        
        
    def get_params(self):
        return self.W, self.b
    
    