import pandas as pd
import numpy as np
import seaborn as sns

# Insert the data
input_data= pd.read_csv('ADULTERED_TRAIN.csv',sep=',')
input_data_test=pd.read_csv('ADULTERED_TEST_NOCLASSES.csv',sep=',')
# Repace the ? with np.nan
input_data.replace(to_replace='[    ]*\?', value=np.nan, regex=True, inplace=True)
input_data_test.replace(to_replace='[    ]*\?', value=np.nan, regex=True, inplace=True)

#Fill in missing values and convert the data into float64 or int values
#for train data
input_data['Engine_Ld_perc_p1'] = pd.to_numeric(input_data['Engine_Ld_perc_p1'].fillna(0))
input_data['Fuel_Trim_Bank1_LT_perc_p1'] = pd.to_numeric(input_data['Fuel_Trim_Bank1_LT_perc_p1'].fillna(0))
input_data['Fuel_Trim_Bank1_ST_perc_p1'] = pd.to_numeric(input_data['Fuel_Trim_Bank1_ST_perc_p1'].fillna(0))
input_data['Fuel_Trim_Bank1_Sensor1_perc_p1'] = pd.to_numeric(input_data['Fuel_Trim_Bank1_Sensor1_perc_p1'].fillna(0))
input_data['O2_Bank1_Sensor1_Volt_p1'] = pd.to_numeric(input_data['O2_Bank1_Sensor1_Volt_p1'].fillna(0))
input_data['Abs_Throttle_Pos_B_perc_p1'] = pd.to_numeric(input_data['Abs_Throttle_Pos_B_perc_p1'].fillna(0))
input_data['Rel_Throttle_Pos_perc_p1'] = pd.to_numeric(input_data['Rel_Throttle_Pos_perc_p1'].fillna(0))
input_data['Engine_RPM_p1'] = pd.to_numeric(input_data['Engine_RPM_p1'].fillna(0))
input_data['Intake_Air_Temp_C_p1'] = pd.to_numeric(input_data['Intake_Air_Temp_C_p1'].fillna(0))
input_data['Engine_Coolant_Temp_C_p1'] = pd.to_numeric(input_data['Engine_Coolant_Temp_C_p1'].fillna(0))
#for test data
input_data_test['Engine_Ld_perc_p1'] = pd.to_numeric(input_data_test['Engine_Ld_perc_p1'].fillna(0))
input_data_test['Fuel_Trim_Bank1_LT_perc_p1'] = pd.to_numeric(input_data_test['Fuel_Trim_Bank1_LT_perc_p1'].fillna(0))
input_data_test['Fuel_Trim_Bank1_ST_perc_p1'] = pd.to_numeric(input_data_test['Fuel_Trim_Bank1_ST_perc_p1'].fillna(0))
input_data_test['Fuel_Trim_Bank1_Sensor1_perc_p1'] = pd.to_numeric(input_data_test['Fuel_Trim_Bank1_Sensor1_perc_p1'].fillna(0))
input_data_test['O2_Bank1_Sensor1_Volt_p1'] = pd.to_numeric(input_data_test['O2_Bank1_Sensor1_Volt_p1'].fillna(0))
input_data_test['Abs_Throttle_Pos_B_perc_p1'] = pd.to_numeric(input_data_test['Abs_Throttle_Pos_B_perc_p1'].fillna(0))
input_data_test['Rel_Throttle_Pos_perc_p1'] = pd.to_numeric(input_data_test['Rel_Throttle_Pos_perc_p1'].fillna(0))
input_data_test['Engine_RPM_p1'] = pd.to_numeric(input_data_test['Engine_RPM_p1'].fillna(0))
input_data_test['Intake_Air_Temp_C_p1'] = pd.to_numeric(input_data_test['Intake_Air_Temp_C_p1'].fillna(0))
input_data_test['Engine_Coolant_Temp_C_p1'] = pd.to_numeric(input_data_test['Engine_Coolant_Temp_C_p1'].fillna(0))
#Split the data into X and y
X=input_data.drop('H2O',axis=1).to_numpy()
y=input_data['H2O']
print(X.shape)
print(y.shape)
#print(input_data_test.info())
print(input_data_test.shape)
#print(input_data.info())

# Activation Functions available for use:
# Relu
def relu(x, derivative=False):
    """
    relu activation function with derivative as Falso
    :param x: np array for each node
    :param derivative: True if we want to use the derivative of this function
    :return: the maximum of 0 or x value and if the derivative is True 1 or 0
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)
#Gelu
def gelu(x, derivative=False):
    """
    gelu activation function
    :param x: np array of the imputs in each node
    :param derivative: True if we want to use the derivative of this function
    :return: the value of gelu function depending if derivative is True on False
    """
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    c = 0.044715
    tanh_arg = sqrt_2_over_pi * (x + c * x ** 3)
    phi_x = 0.5 * (1 + np.tanh(tanh_arg))  # Approximation of Gaussian CDF
    if not derivative:
        return x * phi_x  # GELU function value
    # Compute derivative
    sech2 = 1 / np.cosh(tanh_arg) ** 2  # sech^2(x) = 1 / cosh^2(x)
    phi_derivative = 0.5 * sqrt_2_over_pi * (1 + 3 * c * x ** 2) * sech2
    return phi_x + x * phi_derivative  # GELU derivative
#Leaky relu
def leaky_relu(x, derivative=False, alpha=0.01):
    """
    leaky_relu activation function
    :param x: np array for each node imput
    :param derivative: True for the derivative of leaky_relu
    :param alpha: The parameter of leaky relu
    :return: the output of each node of leaky_relu
    """
    if derivative:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)
#Sigmoid
def sigmoid(x, derivative=False):
    """
    sigmoid activation function for given layer
    :param x: np array of input values in each layer
    :param derivative: if we want to use the derivative of sigmoid activation function
    :return: the output value of simgoid activation function for given layer
    """
    s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    if derivative:
        return s * (1 - s)
    return s
#Tanh
def tanh(x, derivative=False):
    """
    tangent activation function for given layer
    :param x: np array of input values
    :param derivative: True if we want to use the derivative of tanh function
    :return: output of tanh function given particular values
    """
    t = np.tanh(x)
    if derivative:
        return 1 - t**2
    return t
#Linear
def linear(x, derivative=False):
    """
    linear activation function fo given layer
    :param x: np array of input values for each node
    :param derivative: if we want to use the derivative of linear function
    :return: output of linear function
    """
    if derivative:
        return 1
    return x

activation_functions = {'relu': relu,'gelu': gelu,'leaky_relu': leaky_relu,'sigmoid': sigmoid,'tangent': tanh,'linear': linear}
# Get topology from file
class TopologyParser:
    #takes as input file_path where the file is and returns the topology of the file
    def __init__(self, file_path):
        """
        initialization of the class
        :param file_path: the path of the topology file
        """
        self.file_path = file_path
        self.hidden_layers = 0
        self.hidden_nodes = []
        self.activation_functions = []
        self.learning_rate=0
        self.beta1=0
        self.beta2=0
        self.epsilon=0
        self.loss_function=[]
    def parse_file(self): #parses throught the file line by line and reads values based on '='
        """
        function to parse the file and get the values the topology file
        :return: the values for each line of topology file
        """
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('hidden_layers'): # if line starts with hidden_layers we update the values into topology
                self.hidden_layers = int(line.split('=')[1].strip())
            elif line.startswith('hidden_nodes'):
                self.hidden_nodes = list(map(int, line.split('=')[1].strip().split(',')))
            elif line.startswith('activation_functions'):
                self.activation_functions = line.split('=')[1].strip().split(',')
            elif line.startswith('learning_rate'):
                self.learning_rate=float(line.split('=')[1].strip())
            elif line.startswith('beta1'):
                self.beta1=float(line.split('=')[1].strip())
            elif line.startswith('beta2'):
                self.beta2=float(line.split('=')[1].strip())
            elif line.startswith('epsilon'):
                self.epsilon=float(line.split('=')[1].strip())
            elif line.startswith('loss_function'):   # the values must be 'mse' or 'cross_entropy'
                self.loss_function=line.split('=')[1].strip()
        # Validation if there is any inconsistancy in the number of layers and activation functions
        if len(self.hidden_nodes) != self.hidden_layers:
            raise ValueError("Number of hidden nodes doesn't match the number of hidden layers!")
        if len(self.activation_functions) != self.hidden_layers:
            raise ValueError("Number of activation functions doesn't match the number of hidden layers!")
    def get_topology(self):
        """
        function to get the topology of neural network
        :return: the neural network architecture
        """
        return {
            'hidden_layers': self.hidden_layers,
            'hidden_nodes': self.hidden_nodes,
            'activation_functions': self.activation_functions,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,'beta2':self.beta2,'epsion':self.epsilon,'loss_function':self.loss_function
        }
#Create TopologyParser class with topology file name: 'topology.txt'
parser = TopologyParser("topology.txt")
parser.parse_file() #parse the file that reads each line and gets their values
topology = parser.get_topology() #get file content as dictionary
learning_rate=topology['learning_rate'] # learngin rete for nn
beta1=topology['beta1'] # adam optimizer beta1
beta2=topology['beta2'] # adam optimizer beta2
epsilon=topology['epsion'] # adam optimizer epsion
loss_function=topology['loss_function'] #get the loss function from topology
print(topology)
layers=[20]   # 1st layer will have relu activation function and the output layer will have linear ac.f
for i in topology['hidden_nodes']:
    layers.append(i)
layers.append(3)
print(layers)
act_func=['relu']
for i in topology['activation_functions']:
    act_func.append(i)
act_func.append('linear')
print(act_func)

init_method='random' #initial methods to set initial weights
dropout_rate=0.5 # Dropout rate that may be used in traing.

#Initialize weights function
def he_uniform(fan_in, fan_out):
    """
    uniform distribution function the initialize the weights
    :param fan_in: the first size of matrix
    :param fan_out: the second size value of matrix for each weights
    :return: the output matrix of uniformaly distributed weight values
    """
    limit = np.sqrt(2 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))
#dropout function if used.
def dropout(x, dropout_rate):
    """
    the dropout function to be use in traing of neural network
    :param x: np array of input values for each node
    :param dropout_rate: the rate we want to dropout the node values
    :return: np array with droped values
    """
    #Applies dropout by randomly setting some activations to zero
    if dropout_rate > 0:
        mask = (np.random.rand(*x.shape) > dropout_rate) / (1 - dropout_rate)
        return x * mask
    return x
#Cross entropy loss function for the last layer.
def cross_entropy(y_true, y_pred, derivative=False):
    """
    cross entropy loss function for the last layer
    :param y_true: the actual values of target column
    :param y_pred: the predicted values of target column
    :param derivative: if we want to use the derivative for backpropagation
    :return: the loss value  of predicted and the actual values.
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    if derivative:
        return y_pred - y_true  # Derivative w.r.t logits
    # Compute cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)  # Return mean loss over batch
# MSE loss function
def mse_loss(y_true, y_pred, derivative=False):
    """
    MSE loss function for the last layer
    :param y_true: the actual values of target column
    :param y_pred: the predicted values fo target column
    :param derivative: if we want to use the derivative of mse loss
    :return: the loss value of predicted and actual values
    """
    if derivative:
        return 2 * (y_pred - y_true) / y_true.shape[0]  # Derivative w.r.t predictions
    # Compute MSE loss
    loss = np.mean((y_true - y_pred) ** 2)
    return loss  # Return mean loss
#Initiate the values based on the Neural network architecture:
input_num=20  # we always have 20 nodes in the 1st layer
init_method='random' # use random method to initialize the weights
layers  # Number of layers in NN
#for each layer we have this list that keeps the values of NN.
layers2=[] # list of dict values where 'weights': [weights of NN], 'biases':[baises of NN],'activation':[activation func]
for i in range(len(layers)):
    if i==0:
        weights=np.random.randn(input_num,layers[i])
    else:
        weights=np.random.randn(layers[i-1],layers[i])
    layers2.append({'weights':weights,'biases':np.zeros(layers[i]),'activation':act_func[i]})

# Shuffle data and train test split it to train the NN on X_train and y_train and evaluated on X_test.
def train_test_split(X, y, test_size=0.2, seed=42):
    """
    train test split function for training the dataset
    :param X: The entire dataset of csv file
    :param y: the target column
    :param test_size: test size to split the data
    :param seed: random seed
    :return: X_train,X_test,y_train,y_test data
    """
    np.random.seed(seed)  # Ensure reproducibility
    indices = np.arange(X.shape[0])  # Create index array
    np.random.shuffle(indices)  # Shuffle indices

    split_idx = int(X.shape[0] * (1 - test_size))  # Compute split index

    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test
# Split the data into X_train,y_train.. on 20/80.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Convert the y_test values with [0,2.5,5] into array of shape (400,3) same as one-hot encoding
#the same will be done on the y_train data.
class_labels = np.array([0, 2.5, 5])
y_test = np.eye(len(class_labels))[np.searchsorted(class_labels, y_test)]

# Feed forwardpass through network
def feedforwardpass(data):
    """
    feed forwardpass through the neural network
    :param data: values of np.arrey that we use
    :return: the output of neural network and list of weights, biases and activation functions in each layer
    """
    # input data/np.arrey that goes throught the nn.
    # returns: output of the Feed forward pass,the list of weights,biases and activation functions in each layer.
    X=data
    cache2=[] # we cache the values of each weights,biases and activation functions as they pass throught NN.
    for i,layer in enumerate(layers2):
        if i==0:
            w=layer['weights'] #gets the weights for 1st layer
            b=layer['biases']  #gets the biases for 1st layer
            act_f=layer['activation'] #get the acti.fun for 1st layer
            z=np.dot(X,w)+b  # caculate the dot product of weights and input data + biases
            a=activation_functions[act_f](z) # apply activation function on 1st layer on the dot product
            cache2.append((z,a,w,act_f))  # append all the values into cache2 list for backpropagation.
        else:
            w=layer['weights'] # same as before but for the rest of the layers.
            b=layer['biases']
            act_f=layer['activation']
            z=np.dot(cache2[i-1][1],w)
            a=activation_functions[act_f](z)
            cache2.append((z,a,w,act_f))
    return a,cache2 #output of the NN , the cache2 list of dictionaries.

# One-Hot encoding for y_train column
class_labels = np.array([0, 2.5, 5])
y_true = np.eye(len(class_labels))[np.searchsorted(class_labels, y_train)]

# Backpropagation fucntion with Adam optimizer to train on ff-nn
# Inputs:
# -the cache2 list of weights,biases and act.functions for each layer in NN.
# - X : data that is used in feetfarward
# - y_true: the one-hot encoded arrey of target column
# - loss_type: loss function 'cross_entropy' or 'mse'
# - Adam optimizer parametes if changes from topolgy file
# returns: the backpropagaion2 function just updates the values of layers2 weights,biases using adam optimizer
def backpropagation2(cache2, X, y_true, loss_type='cross_entropy', learning_rate=0.001, beta1=0.9, beta2=0.999,
                     epsilon=1e-8):
    """
    backpropagation function that updated weights and biases based on adam optimizer
    :param cache2: the values of neural netwrok such as weights biaes and activation functions
    :param X: The entier dataset
    :param y_true: the target column
    :param loss_type: loss function type for last layer
    :param learning_rate: learning rate for adam optimizer
    :param beta1: beta1 value for adam optimizer
    :param beta2: beta2 value for adam optimizer
    :param epsilon: epsilon value for adam optimizer
    :return: None just updates the values in layers2 list for neural network based on adam optimizer
    """
    m = [np.zeros_like(layer['weights']) for layer in layers2] #initialize adams first-moment vector
    v = [np.zeros_like(layer['weights']) for layer in layers2] # initialize adams second-moment vector
    m_biases = [np.zeros_like(layer['biases']) for layer in layers2] # initialize biases
    v_biases = [np.zeros_like(layer['biases']) for layer in layers2]
    t = 0 # inital moment
    # compute initial gradint from loss function
    y_pred = cache2[-1][1] #get the output of FF pass
    if loss_type == 'cross_entropy': #if loss function is cross_entropy
        dz = y_pred - y_true
    else:  # MSE loss function
        dz = 2 * (y_pred - y_true) / y_true.shape[0]
    # in revers order go back to front in nn
    for i in reversed(range(len(layers))):
        # get all the values for particular layer: input in the layer,ouput of layer,weights of that layer,act funtion.
        z, a, w, activation = cache2[i]
        da = activation_functions[activation](z, derivative=True) * dz #gradient of ac.function times error
        dw = np.dot(cache2[i - 1][1].T, da) if i > 0 else np.dot(cache2[i][1].T, da) # derivaties for each weight
        # print('da',da.shape)
        dz = np.dot(da, w.T) # dot product of weights and act functions
        # print(dz.shape)
        db = np.sum(da, axis=0) / y_true.shape[0] # derivates of biases
        # adam optimizer updates
        t += 1
        m[i] = beta1 * m[i] + (1 - beta1) * dw        # update first-moment vectors
        v[i] = beta2 * v[i] + (1 - beta2) * (dw ** 2) # update second-moment vectors
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * db  #update biases
        v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * (db ** 2)
        m_hat_b = m_biases[i] / (1 - beta1 ** (t + 1))
        v_hat_b = v_biases[i] / (1 - beta2 ** (t + 1))

        layers2[i]['weights'] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)    #update the parameters based on adam
        layers2[i]['biases'] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

#MSE loss function.
#def compute_loss(y_true, y_pred):
#    return np.mean((y_true - y_pred) ** 2)

#Given a NumPy array with shape (n, 3), finds the maximum value in each row
#sets it to 1, and sets the rest to 0.
def binarize_max(arr: np.ndarray) -> np.ndarray:
    """
    function to convert the max value in a array into 1 and the rest to 0
    :param arr: NumPy array of shape (n, 3)
    :return: Binarized NumPy array of the same shape
    """
    if arr.shape[1] != 3:
        raise ValueError("Input array must have shape (n, 3)")
    max_indices = np.argmax(arr, axis=1)  # Get indices of max values per row
    binarized = np.zeros_like(arr)  # Initialize array of zeros
    binarized[np.arange(arr.shape[0]), max_indices] = 1  # Set max positions to 1
    return binarized

#Train the NN with batches of size 32.
# input: - X:data for training, -y:target values, -epochs: number of iterations, batch_size
# output: Every 10 epoches print the test Accuracy and Loss value.
def train(X, y, epochs=1000, batch_size=32,loss_type='cross_entropy'):
    """
    train funtion for neural network
    :param X: the entier dataset we are training on
    :param y: the target values we trying to predict
    :param epochs: epiche size for the train
    :param batch_size: batch size for each matrix
    :param loss_type: loss type function for last layer
    :return: the accuracy of test set and the loss value for each epoche
    """
    X_train,X_test,y_train,y_test=train_test_split(X, y)
    class_labels = np.array([0, 2.5, 5])
    y_true = np.eye(len(class_labels))[np.searchsorted(class_labels, y_train)]
    y_test = np.eye(len(class_labels))[np.searchsorted(class_labels, y_test)]
    #
    for epoch in range(epochs):
        #y_true = np.eye(len(class_labels))[np.searchsorted(class_labels, y)]
        for i in range(0, X_train.shape[0]-31, batch_size):
            X_batch = X_train[i:i+batch_size] #split the data into batches
            y_batch = y_true[i:i+batch_size]
            #print(y_batch.shape)
            output,cache=feedforwardpass(X_batch)  # feedforwardpass through NN
            backpropagation2(cache,X_batch, y_batch,loss_type=loss_function) #backpropagate with adam
        if epoch % 10 == 0:
            if loss_type=='cross_entropy':
                loss=cross_entropy(y_batch,output) #calculate the loss w
            else:
                loss = mse_loss(y_batch, output)
            print(f"Epoch {epoch}, Loss: {loss:.4f}") #print for each 10 epoch the loss value.
            # predict the value of each batch
            def predict(X):
                output,cache2=feedforwardpass(X)
                return output
            def accuracy_score(y_test, y_pred): #accuracy score function
                return np.mean(y_test == y_pred)
            y_pred = predict(X_test)
            y_p_class=binarize_max(y_pred)
            # Compute accuracy and print it
            accuracy = accuracy_score(y_test, y_p_class)*100
            print(f"Test Accuracy: {accuracy:.4f}%")

#see the accuracy before training and after training:
train(X,y) #train the model for 1000 epoches
#Predict the test datasets values
X_TEST_PRED=input_data_test.to_numpy()
pred,cache3=feedforwardpass(X_TEST_PRED)
p=binarize_max(pred)
#print(binarize_max(pred))
#print(len(p))
pred_values=[]
for i in range(len(p)):
    for j in range(3):
        if j==0 and p[i][j]==1:
            pred_values.append(0)
        elif j==1 and p[i][j]==1:
            pred_values.append(2.5)
        elif j==2 and p[i][j]==1:
            pred_values.append(5)
print('the prediction on TEST dataset:')
print(pred_values)
print(len(pred_values))