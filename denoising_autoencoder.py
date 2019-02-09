import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import random
#from PIL import Image


data_dir = r'C:\Users\Welcome\Documents\code_python_shelll\fashion_mnist'


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
    # print("lin_cache=",lin_cache["A"].shape)

    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def denoising_autoencoder(n_in, n_h, n_fin, train,tsX, trX, trX_copy, learning_rate, decay_rate):

    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    A0 = train
    costs = []
    idx = []

    for ii in range(100):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]
        #FORWARD
        A1, cache1 = layer_forward(A0, W1, b1, "sigmoid")
        A2, cache2 = layer_forward(A1, W2, b2, "sigmoid")

        #COST
        cost = cost_estimate(A2, trX)

        m = trX_copy.shape[1]
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
         #   print("Cost of Validation set at iteration %i is: %f" % (ii, cost_val))

    A1, cache1 = layer_forward(trX, parameters["W1"], parameters["b1"], "sigmoid")
    A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

    rows = 1
    columns = 10
    fig=plt.figure(figsize=(80, 80))
    #j = 0
    for i in range(10):
        #print(A2.shape)
        act = trX[:,1+100*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(act.reshape(28, -1),cmap="Greys")
    plt.show()
    fig=plt.figure(figsize=(80, 80))
    for i in range(10):   
        noisy = train[:,1+100*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(noisy.reshape(28, -1),cmap="Greys")
    #plt.title(' corrupted Images from 10 different classes')
    plt.show()
    fig=plt.figure(figsize=(50, 50))
    for i in range(10):    
        test = A2[:,1+100*i]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(test.reshape(28, -1),cmap="Greys")
    #plt.title('Reconstructed Images from 10 different classes')
    plt.show()

def main():
    no_of_Tr_Samaples = 1000
    no_of_Ts_Samaples = 100
    digit_rng = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    no_of_Tr_Per_Class = 100
    no_of_Ts_Per_Class = 10
    trX, trY, tsX, tsY = mnist(noTrSamples=no_of_Tr_Samaples,
                               noTsSamples=no_of_Ts_Samaples, digit_range=digit_rng,
                               noTrPerClass=no_of_Tr_Per_Class, noTsPerClass=no_of_Ts_Per_Class)

    train = np.zeros((no_of_Tr_Samaples,784))
    trX_copy = np.zeros((no_of_Tr_Samaples,784))
    trX_copy = trX.copy()
    noise_level = 20
    c = range(0,784)
    noise = int((784*noise_level)/100)
    mask = random.sample(c,noise)
    for i in range(trX.shape[1]):
        img = trX_copy[:,i]
        #mask = np.random.randint(0, 784, 50)
        for m in mask:
           img[m] = 0
        train[i] = img
    train = train.T

    n_in = 784
    n_h  = 1400
    n_fin = 784
    learning_rate = 0.1
    decay_rate = 0

    denoising_autoencoder(n_in, n_h, n_fin, train,tsX,trX, trX_copy, learning_rate, decay_rate)


if __name__ == "__main__":
    main()
