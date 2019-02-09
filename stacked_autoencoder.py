import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import random
#from PIL import Image
epsilon = 1e-5

data_dir = r'C:/Users/Welcome/Documents/code_python_shelll/fashion_mnist'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10): #, noTvSamples= 400, noTvPerClass=200):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    #assert noTvSamples == noTvPerClass * len(digit_range), 'noTvSamples and noTvPerClass mismatch'
    #data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)


    return trX, trY, tsX, tsY


def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs:
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE

    W1 = np.random.randn(n_h, n_in) * np.sqrt(1 / (n_in + n_h))
  #  W2 = np.random.randn(n_fin, n_h) * np.sqrt(1 / (n_fin + n_h))
    W2 = W1.T
    b1 = np.random.randn(n_h, 1) * 0.01
    b2 = np.random.randn(n_in, 1) * 0.01

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters


def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z = cache["Z"]
    #  A2, cache = sigmoid(cache["Z"])
    A2 = 1 / (1 + np.exp(-Z))
    dZ = dA * A2 * (1 - A2)
    return dZ

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z = cache["Z"]
    # tanh_Z, cache = tanh(Z)
    dZ = 1.0 - np.tanh(Z) ** 2
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE 
    A = np.exp(Z - np.max(Z))
    sums = A.sum(axis=0)
    np.set_printoptions(precision=100)
    for index, x in np.ndenumerate(A):
        A[index[0], index[1]] = x/sums[index[1]]
    cache = {}
    cache['A'] = A

    if Y.shape[0]:
        cross_entropy_array = []
        for index, x in np.ndenumerate(Y):
            cross_entropy_array.append(-1. * np.log(A[int(x), index[1]] + epsilon))
        cross_entropy = np.asarray(cross_entropy_array).reshape(1, Y.shape[1])
        loss = np.mean(cross_entropy)
    else:
        loss = None

    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE 
    dZ = cache['A'].copy()
    for index, x in np.ndenumerate(Y):
        dZ[int(x), index[1]] -= 1
    m = Y.shape[1]
    return dZ/m

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer
    Z = WA + b is the output of this layer.

    Inputs:
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    cache["W"] = W
    cache["b"] = b
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs:
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache


def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels

    Returns:
        cost of the objective function
    '''
    ### CODE HERE
    m = Y.shape[1]
    # print(Y.shape[1])
    # cost = (-1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost =  (1 / m ) * np.sum((A2-Y) ** 2)
    #  cost = -(1/1200) * ( np.sum( np.multiply(np.log(A2),Y) ) + np.sum( np.multiply(np.log(1-A2),(1-Y)) ) )
    #   print(cost)
    return cost


def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    dZ2W2 = cache["A"].T

    dW = np.dot(dZ, dZ2W2)
    db = np.sum(dZ, axis=1, keepdims=True)
    # if W.shape[1] != dZ.shape[0]:
    #  W = W.T

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]  # A
    act_cache = cache["act_cache"]  # Z

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    # print("lin_cache=",lin_cache["A"].shape)

    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def softmax_classifier_layer(n_in, n_h, train, Y, learning_rate, decay_rate):
    parameters = initialize_2layer_weights(n_in, n_h, n_in)
    A0 = train
    costs = []
    idx = []

    for ii in range(1000):
        W1 = parameters["W1"]
        # W2 = parameters["W2"]
        b1 = parameters["b1"]
        # b2 = parameters["b2"]
        #FORWARD
        A1, cache1 = layer_forward(A0, W1, b1, "linear")
        # A2, cache2 = layer_forward(A1, W2, b2, "linear")

        A2, cache2, cost = softmax_cross_entropy_loss(A1, Y)

        # m = Y.shape[1]
        # dA1 = -1/m * ((Y/A2) - (Y-1)/(A2-1))
        dZ = softmax_cross_entropy_loss_der(Y, cache2)

        # dA1, dW2, db2 = layer_backward(dA2, cache2, W2, b2, "linear")
        dA0, dW1, db1 = layer_backward(dZ, cache1, W1, b1, "linear")


        #update parameters
        ### CODE HERE
        alpha = learning_rate * (1 / (1 + decay_rate * ii))
        parameters["W1"] = W1 - (alpha * dW1)
        parameters["b1"] = b1 - (alpha * db1)
        # parameters["W2"] = W2 - (alpha * dW2)
        # parameters["b2"] = b2 - (alpha * db2)

        if ii % 10 == 0:
            costs.append(cost)
            idx.append(ii)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
         #   print("Cost of Validation set at iteration %i is: %f" % (ii, cost_val))
    return parameters


def denoising_autoencoder(n_in, n_h, train, trX, learning_rate, decay_rate):

    parameters = initialize_2layer_weights(n_in, n_h, n_in)
    A0 = train # this is trX since i am passing trX in train
    costs = []
    idx = []
    final_cost = 0
    for ii in range(1000):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        #FORWARD
        A1, cache1 = layer_forward(A0, W1, b1, "sigmoid")
        A2, cache2 = layer_forward(A1, W2, b2, "sigmoid")

        #COST
        cost = cost_estimate(A2, trX)

        m = trX.shape[1]
        dA2 = -1/m * ((trX/A2) - (trX-1)/(A2-1))


        dA1, dW2, db2 = layer_backward(dA2, cache2, W2, b2, "sigmoid")
        dA0, dW1, db1 = layer_backward(dA1, cache1, W1, b1, "sigmoid")


        #update parameters
        ### CODE HERE
        alpha = learning_rate * (1 / (1 + decay_rate * ii))
        parameters["W1"] = W1 - (alpha * dW1)
        parameters["b1"] = b1 - (alpha * db1)
        parameters["W2"] = W2 - (alpha * dW2)
        parameters["b2"] = b2 - (alpha * db2)

        if ii % 10 == 0:
            costs.append(cost)
            idx.append(ii)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
        final_cost = cost
         #   print("Cost of Validation set at iteration %i is: %f" % (ii, cost_val))


    A1, cache1 = layer_forward(train, parameters["W1"], parameters["b1"], "sigmoid")
    A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

    # if (n_in == 784):
    rows = 1
    columns = 10
    fig=plt.figure(figsize=(20, 20))
    #j = 0
    for i in range(10):
        #print(A2.shape)
        act = trX[:,5+500*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(act.reshape(28, -1),cmap="Greys")
    plt.show()
    fig=plt.figure(figsize=(20, 20))
    for i in range(10):   
        noisy = train[:,5+500*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(noisy.reshape(28, -1),cmap="Greys")
    #plt.title(' corrupted Images from 10 different classes')
    plt.show()
    fig=plt.figure(figsize=(20, 20))
    for i in range(10):    
        test = A2[:,5+500*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(test.reshape(28, -1),cmap="Greys")
    #plt.title('Reconstructed Images from 10 different classes')
    plt.show()
            # test = A2[:,6+i*10]
            # plt.imshow(test.reshape(28, -1), cmap="Greys")
            # plt.show()
    return A1, parameters, final_cost

def classify(tsX, tsY, h1_parameters, h2_parameters, h3_parameters, h4_parameters):
    A1, cache1 = layer_forward(tsX, h1_parameters["W1"], h1_parameters["b1"], "sigmoid")
    A2, cache1 = layer_forward(A1, h2_parameters["W1"], h2_parameters["b1"], "sigmoid")
    A3, cache1 = layer_forward(A2, h3_parameters["W1"], h3_parameters["b1"], "sigmoid")
    A4, cache1 = layer_forward(A3, h4_parameters["W1"], h4_parameters["b1"], "linear")
    A, _, _ = softmax_cross_entropy_loss(A4)
    labels = np.argmax(A, axis=0)
    Ypred = labels.reshape(1, len(labels))
    return Ypred

def get_noise_data(no_of_Tr_Samaples, noise_level, trX ):
    train = np.zeros((no_of_Tr_Samaples,784))
    trX_copy = np.zeros((no_of_Tr_Samaples,784))
    trX_copy = trX.copy()
    c = range(0, 784)
    noise = int((784 * noise_level)/100)
    mask = random.sample(c, noise)
    for i in range(trX.shape[1]):
        img = trX_copy[:,i]
        for m in mask:
           img[m] = 0
        train[i] = img
    train = train.T
    return train


def stacked_autoencoder():
    no_of_Tr_Samaples = 5000
    no_of_Ts_Samaples = 5000
    digit_rng = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    no_of_Tr_Per_Class = 500
    no_of_Ts_Per_Class = 500
    n_in = 784
    n_h_1  = 500
    n_h_2 = 200
    n_h_3 = 100
    n_fin = 10
    learning_rate = 0.1
    decay_rate = 0
    noise_level = 20

    trX, trY, tsX, tsY = mnist(noTrSamples=no_of_Tr_Samaples,
                               noTsSamples=no_of_Ts_Samaples, digit_range=digit_rng,
                               noTrPerClass=no_of_Tr_Per_Class, noTsPerClass=no_of_Ts_Per_Class)

    #train = get_noise_data(no_of_Tr_Samaples, noise_level, trX ) #Not going to use this use trX
    #test = get_noise_data(no_of_Ts_Samaples, noise_level, tsX )  # Not going to use this use tsX

    h1_A1, h1_parameters,temp_cost1 = denoising_autoencoder(n_in, n_h_1, trX, trX, learning_rate, decay_rate)
    h2_A1, h2_parameters,temp_cost2 = denoising_autoencoder(n_h_1, n_h_2, h1_A1, h1_A1, learning_rate, decay_rate)
    h3_A1, h3_parameters,temp_cost3 = denoising_autoencoder(n_h_2, n_h_3, h2_A1, h2_A1, learning_rate, decay_rate)

    no_of_labels_per_class = 100
    
    trX_class, trY_class, tsX_class, tsY_class = mnist(noTrSamples = no_of_labels_per_class*10,
                               noTsSamples=no_of_Ts_Samaples, digit_range=digit_rng,
                               noTrPerClass=no_of_labels_per_class, noTsPerClass=no_of_Ts_Per_Class)
    #train_trX_class = get_noise_data(no_of_labels_per_class*10, noise_level, trX_class )
    A1, cache1 = layer_forward(trX_class, h1_parameters["W1"], h1_parameters["b1"], "sigmoid") #Giving raw data
    A2, cache1 = layer_forward(A1, h2_parameters["W1"], h2_parameters["b1"], "sigmoid")
    A3, cache1 = layer_forward(A2, h3_parameters["W1"], h3_parameters["b1"], "sigmoid")

    h4_parameters = softmax_classifier_layer(n_h_3, n_fin, A3, trY_class, learning_rate, decay_rate)

    test_Pred = classify(tsX, tsY, h1_parameters, h2_parameters, h3_parameters, h4_parameters)
    train_Pred = classify(trX, trY, h1_parameters, h2_parameters, h3_parameters, h4_parameters)

    count_train_errors = 0
    count_test_errors = 0
    # print(test_label)
    # print(test_Pred)
    for i in range(len(train_Pred[0])):
        if train_Pred[0][i] != trY[0][i] :
            count_train_errors = count_train_errors + 1
    for j in range(len(test_Pred[0])):
        if test_Pred[0][j] != tsY[0][j] :
            count_test_errors = count_test_errors + 1

    print(count_train_errors)
    print(count_test_errors)
    trAcc = ((len(train_Pred[0]) - count_train_errors) * 100)/ len(train_Pred[0])
    teAcc = ((len(test_Pred[0]) - count_test_errors) * 100)/ len(test_Pred[0])
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

def main():
    no_of_Tr_Samaples = 5000
    no_of_Ts_Samaples = 50
    digit_rng = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    no_of_Tr_Per_Class = 500
    no_of_Ts_Per_Class = 5
    n_in = 784
    n_h  = 1000
    n_fin = 784
    learning_rate = 0.1
    decay_rate = 0
    noise_level = 20

    trX, trY, tsX, tsY = mnist(noTrSamples=no_of_Tr_Samaples,
                               noTsSamples=no_of_Ts_Samaples, digit_range=digit_rng,
                               noTrPerClass=no_of_Tr_Per_Class, noTsPerClass=no_of_Ts_Per_Class)

    train = get_noise_data(no_of_Tr_Samaples, noise_level, trX )

    # varying learning rate
    # costs = []
    # x = []
    # for i in range(1, 11):
    #     A1, parameters, cost = denoising_autoencoder(n_in, n_h, train, trX, (learning_rate*i)/2, decay_rate)
    #     x.append((learning_rate*i)/2)
    #     costs.append(cost)
    #     print("Final Cost at %i is: %.08f" %(i, cost))
    # plt.plot(x,costs, 'r')
    # plt.xlabel('learning rate')
    # plt.ylabel('Cost after 500 iterations')
    # plt.title('Cost vs Learning rate')
    # plt.show()

    #varying nodes
    '''costs = []
    x = []
    for i in range(11):
        A1, parameters, cost = denoising_autoencoder(n_in, (500 + (i*50)), train, trX, learning_rate, decay_rate)
        x.append(500 + (i*50))
        costs.append(cost)
    plt.plot(x,costs, 'r')
        print("Final Cost at %i is: %.08f" %(i, cost))
    plt.xlabel('Hidden Nodes')
    plt.ylabel('Cost after 500 iterations')
    plt.title('Cost vs Hidden Nodes plot')
    plt.show()'''


    #stacked auto-encoder
    #stacked_autoencoder()
    denoising_autoencoder(n_in, n_h, train, trX, learning_rate, decay_rate)

if __name__ == "__main__":
    main()
