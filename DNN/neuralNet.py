import numpy as np
from testCases_v2 import *

def initialize_parameters(n_x=2, n_h=5, n_y=1):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))


    ### END CODE HERE ###
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}

    return parameters





def initialize_parameters_deep(layer_dims=[3, 5, 1]):
    """
    Arguments:
        layer_dims -- python array (list) containing the dimensions of each layerinournetwork
    Returns:
        parameters -- python dictionary containing your parameters "W1","b1",...,"WL","bL":
          Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
          bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters



def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
        A -- activations from previous layer (or input data): (size of previouslayer,numberofexamples)
        W -- weights matrix: numpy array of shape (size of current layer, sizeofpreviouslayer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
        Z -- the input of the activation function, also called pre-activationparameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def sigmoid(Z):
    """
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    """
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation="relu"):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
        A -- the output of the activation function, also called the post-activationvalue
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
    stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def test_linear_activation_forward():
    A_prev = np.random.randn((3,1))
    assert(a.shape == (3,1))
    W, b = initialize_parameters_deep([3, 5]).values()
    linear_activation_forward(A_prev, W, b)



def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOIDcomputation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them,indexedfrom0toL-2)
                the cache of linear_sigmoid_forward() (there is one, indexedL-1)
    """
    caches = []
    A=X
    L = len(parameters) // 2 # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
    ### START CODE HERE ### (  2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)],activation="relu")
        caches.append(cache)
    ### END CODE HERE ###
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list. ### START CODE HERE ### (  2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
    ### END CODE HERE ###
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches



def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).
    Arguments:
        AL -- probability vector corresponding to your label predictions, shape(1,numberofexamples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat),shape(1, number of examples)
    Returns:
        cost -- cross-entropy cost
    """
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = -(np.dot(Y, np.log(AL.T)) + np.dot(1-Y, np.log(1-AL).T))/m
    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect(e.g.this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost



def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer(layerl)
    Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layerl)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db




def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)
    return dZ



def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1)->LINEAR->SIGMOIDgroup
    Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it'scaches[l],forlinrange(L-1)i.el=0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it'scaches[L-1])
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) ### END CODE HERE ###
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches".  → Outputs:"grads["dAL"],grads["dWL"],grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    ### END CODE HERE ###
    for l in reversed(range(L-1)):
    # lth layer: (RELU -> LINEAR) gradients.
    # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA"+str(l+1)],grads["dW"+str(l+1)], grads["db"+str(l+1)]
    ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters
