import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm
import os
import json
import random
from numpy import longdouble

#Initialize helpful functions for math
def sigmoid(x: np.ndarray):
    return 1/(1+np.exp(-1*x))

def sigmoid_derivative(x: np.ndarray):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x: np.ndarray):
    return np.tanh(x)
    
def tanh_derivative(x:np.ndarray):
    return 1-np.square(tanh(x))

def softmax(x: np.ndarray):
    return np.exp(x)/np.sum(np.exp(x))

def cross_entropy(yhat, y, epsilon=1e-10):
    yhat_clipped = np.clip(yhat, epsilon, 1 - epsilon)  # Clip yhat to avoid zeros
    return -np.sum(y * np.log(yhat_clipped))

def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (input_size, output_size)).astype(longdouble) * np.sqrt(6 / (input_size + output_size))


#initializes the weights of the network
def initialize_cell(input_size, hidden_size):
    

    cell = {}

    cell["W_i"] = np.vstack((initWeights(hidden_size, hidden_size), initWeights(input_size, hidden_size))) #input gate weights
    cell["W_f"] = np.vstack((initWeights(hidden_size, hidden_size), initWeights(input_size, hidden_size))) #forget gate weights
    cell["W_c"] = np.vstack((initWeights(hidden_size, hidden_size), initWeights(input_size, hidden_size))) #candidate gate weights
    cell["W_o"] = np.vstack((initWeights(hidden_size, hidden_size), initWeights(input_size, hidden_size))) #output gate weights
    cell["W_y"] = initWeights(hidden_size, 10000)#final gate weights

    #not sure if the biases need to be 3d...
    cell["b_i"] = np.zeros(hidden_size,dtype=longdouble) #input gate biases
    cell["b_f"] = np.zeros(hidden_size,dtype=longdouble) #forget gate biases
    cell["b_c"] = np.zeros(hidden_size,dtype=longdouble) #candidate gate biases
    cell["b_o"] = np.zeros(hidden_size,dtype=longdouble) #output gate biases
    cell["b_y"] = np.zeros(10000) #final gate biases

    return cell

#forward pass of all gates
def forward_pass(cell, prevA, prevC, X):

    # print(X, "/n______-")
    
    # print(X)
    input = np.hstack((prevA, X))
   

    forward = {}
    # print(cell["W_f"])

    forward["F"] = sigmoid(input.dot(cell["W_f"]) + cell["b_f"])

    forward["_c"] = input.dot(cell["W_c"]) + cell["b_c"]
    
    forward["C"] = tanh(forward["_c"])

    forward["I"] = sigmoid(input.dot(cell["W_i"]) + cell["b_i"])

    forward["O"] = sigmoid(input.dot(cell["W_o"]) + cell["b_o"])


    forward["prevA"] = prevA
    forward["prevC"] = prevC
    forward["C_t"] = (forward["prevC"] * forward["F"]) + (forward["I"] * forward["C"])
    forward["A_t"] = forward["O"] * tanh(forward["C_t"])

    forward["Z_t"] = forward["A_t"].dot(cell["W_y"]) 
    # + cell["b_y"]
    
    forward["Yhat"] = softmax(forward["Z_t"])
    # 
    # print(forward["Yhat"].size)
    # print(forward["Yhat"], "  Yhat")
    return forward

def gradient(forward, cell, X, Y, lprimea, lprimec, hidden_size):
    # print("HELLLOOOOOOOOO")
    grads = {}

    # print("BackProp")
    input = np.hstack((forward["prevA"], X))
    # print((forward["Yhat"]-Y).size, "yhat-y")
    # print((np.transpose(cell["W_y"]).size))
    # print(lprimea.size)
    dldA_t = (forward["Yhat"]-Y).dot(np.transpose(cell["W_y"])) + lprimea
    
    dldC_t = lprimec + (forward["O"] * tanh_derivative(forward["C_t"])) * dldA_t 
    # print(forward["Yhat"]-Y)

    TdLdw_f = (dldC_t * forward["prevC"] * forward["F"]*(1-forward["F"])) 
    # TdLdw_c = (dldC_t * forward["I"])
    TdLdw_c = (dldC_t * forward["I"]*tanh_derivative(forward["_c"]))
    TdLdw_o = (dldA_t * tanh(forward["C_t"]) * forward["O"] * (1-forward["O"]))
    TdLdw_i = (dldC_t * forward["C"] * forward["I"] * (1-forward["I"]))
    TdLdw_y = (forward["Yhat"] - Y)

    

    # np.atleast2d(a).T

    woa = cell["W_o"][:hidden_size, :]
    wca = cell["W_c"][:hidden_size, :]
    wia = cell["W_i"][:hidden_size, :]
    wfa = cell["W_f"][:hidden_size, :]


    # print(TdLdw_o.size)
    # print(woa.size)

    grads["dLda_prev"] = TdLdw_o.dot(woa.T) + TdLdw_c.dot(wca.T) + TdLdw_i.dot(wia.T) + TdLdw_f.dot(wfa.T)
    grads["dLdc_prev"] = (lprimec + (forward["O"] * tanh_derivative(forward["C_t"]) * dldA_t)) * forward["F"]


    #not sure which side to transpose.
    grads["dLdw_f"] = np.atleast_2d(input).T.dot(np.atleast_2d(TdLdw_f))
    grads["dLdw_c"] = np.atleast_2d(input).T.dot(np.atleast_2d(TdLdw_c))
    grads["dLdw_o"] = np.atleast_2d(input).T.dot(np.atleast_2d(TdLdw_o))
    grads["dLdw_i"] = np.atleast_2d(input).T.dot(np.atleast_2d(TdLdw_i))
    grads["dLdw_y"] = np.atleast_2d(np.atleast_2d(forward["A_t"])).T.dot(np.atleast_2d(TdLdw_y))

    grads["dLdb_f"] = TdLdw_f.sum(axis=0)
    grads["dLdb_c"] = TdLdw_c.sum(axis=0)
    grads["dLdb_o"] = TdLdw_o.sum(axis=0)
    grads["dLdb_i"] = TdLdw_i.sum(axis=0)
    grads["dLdb_y"] = TdLdw_y.sum(axis=0)


    
    loss = cross_entropy(forward["Yhat"], Y)
    # print(loss)

    return grads, loss

def clip(derivative, norm):
    dernorm = np.linalg.norm(derivative)
    if(dernorm > norm):
        # print("clip")
        derivative = norm * derivative/dernorm

    return derivative

def descent(cell, X, input_size, hidden_size, batch_size, lr, norm):

    # for b in range(0, batch_size):

    prevA = np.zeros((batch_size, hidden_size))
    prevC = np.zeros((batch_size, hidden_size))

    gradientTot = {}
    lossTot = 0

    labels = []

    inputs = []

    allForwards = []

    lprimea = np.zeros((batch_size, hidden_size))
    lprimec = np.zeros((batch_size, hidden_size))

    gradientTot["dLdw_f"] = np.vstack((np.zeros((hidden_size,hidden_size)), np.zeros((input_size, hidden_size))))
    gradientTot["dLdw_c"] = np.vstack((np.zeros((hidden_size,hidden_size)), np.zeros((input_size, hidden_size))))
    gradientTot["dLdw_o"] = np.vstack((np.zeros((hidden_size,hidden_size)), np.zeros((input_size, hidden_size))))
    gradientTot["dLdw_i"] = np.vstack((np.zeros((hidden_size,hidden_size)), np.zeros((input_size, hidden_size))))
    gradientTot["dLdw_y"] = np.zeros((hidden_size, 10000),dtype=longdouble)
    
    gradientTot["dLdb_f"] = np.zeros(hidden_size,dtype=longdouble)
    gradientTot["dLdb_c"] = np.zeros(hidden_size,dtype=longdouble)
    gradientTot["dLdb_o"] = np.zeros(hidden_size,dtype=longdouble)
    gradientTot["dLdb_i"] = np.zeros(hidden_size,dtype=longdouble)
    gradientTot["dLdb_y"] = np.zeros(10000,dtype=longdouble)

    

    for i in range(1, input_size-1):

        blabel = []

        input = np.copy(X)

        for minibatch in input:
            for token in range(i, input_size):
                minibatch[token] = 1

        inputs.append(input)
        # print(len(X))
        for mini in X:
            
            label = np.zeros(10000, dtype=longdouble)
            label[int(mini[i+1])] = longdouble(1)

            blabel.append(label)
            

        forward = forward_pass(cell, prevA, prevC, input)

        prevA = forward["A_t"]
        prevC = forward["C_t"]
    
        labels.append(blabel)
        allForwards.append(forward)

    
    for i in range(0, len(allForwards)):
     
        grad, loss = gradient(allForwards[i], cell, inputs[i], labels[i], lprimea, lprimec, hidden_size)
        lprimea = grad["dLda_prev"]
        lprimec = grad["dLdc_prev"]

        

        gradientTot["dLdw_f"] += grad["dLdw_f"]
        gradientTot["dLdw_c"] += grad["dLdw_c"]
        gradientTot["dLdw_o"] += grad["dLdw_o"]
        gradientTot["dLdw_i"] += grad["dLdw_i"]
        gradientTot["dLdw_y"] += grad["dLdw_y"]
        
        gradientTot["dLdb_f"] += grad["dLdb_f"]
        gradientTot["dLdb_c"] += grad["dLdb_c"]
        gradientTot["dLdb_o"] += grad["dLdb_o"]
        gradientTot["dLdb_i"] += grad["dLdb_i"]
        gradientTot["dLdb_y"] += grad["dLdb_y"]

        lossTot += loss

    gradientTot["dLdw_f"] = clip(gradientTot["dLdw_f"]/batch_size, norm)
    gradientTot["dLdw_c"] = clip(gradientTot["dLdw_c"]/batch_size, norm)
    gradientTot["dLdw_o"] = clip(gradientTot["dLdw_o"]/batch_size, norm)
    gradientTot["dLdw_i"] = clip(gradientTot["dLdw_i"]/batch_size, norm)
    gradientTot["dLdw_y"] = clip(gradientTot["dLdw_y"]/batch_size, norm)

    cell["W_f"] = cell["W_f"] - gradientTot["dLdw_f"] * lr
    cell["W_c"] = cell["W_c"] - gradientTot["dLdw_c"] * lr
    cell["W_o"] = cell["W_o"] - gradientTot["dLdw_o"] * lr
    cell["W_i"] = cell["W_i"] - gradientTot["dLdw_i"] * lr
    cell["W_y"] = cell["W_y"] - gradientTot["dLdw_y"] * lr

    cell["b_f"] = cell["b_f"] - gradientTot["dLdb_f"]/batch_size * lr
    cell["b_c"] = cell["b_c"] - gradientTot["dLdb_c"]/batch_size * lr
    cell["b_o"] = cell["b_o"] - gradientTot["dLdb_o"]/batch_size * lr
    cell["b_i"] = cell["b_i"] - gradientTot["dLdb_i"]/batch_size * lr
    cell["b_y"] = cell["b_y"] - gradientTot["dLdb_y"]/batch_size * lr

    return lossTot


def train(dataset, input_size, hidden_size, batch_size, lr, norm):

    cell = initialize_cell(input_size, hidden_size)

    losses = []

    for i in tqdm(range(0,100)):
        loss = descent(cell, dataset[i], input_size, hidden_size, batch_size, lr, norm)
        print(loss)
        losses.append(loss)


    plt.plot(np.arange(len(losses)) * batch_size, losses)
    plt.title("training curve for LR: ", lr, ", BS: ", batch_size, ", HS: ", hidden_size, " norm: ", norm)
    plt.xlabel("number of emails trained on")
    plt.ylabel("loss")
    plt.show()
    return cell, losses