import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
np.random.seed(5)
X = np.random.rand(1,3)
X -= 0.5
X *= math.pi * 2
Y = 2 * X
plt.scatter(X,Y)
#plt.show()
layer_size = [1,5,1]
layers = len(layer_size)

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z
def sigmoid_grad(x):
    z = 1/(1 + np.exp(-x))
    return (z) * (1-z)

def forward_propagation(params,X,layer_size):
    layers = len(layer_size)
    cache = {"A0":X}
    cA = X
    L = layers-1
    for i in range(1,layers-1):

        current_l = i
        cW = params["W"+str(current_l)]
        cb = params["b"+str(current_l)]
        halfZ = np.dot(cW,cA)
        cZ = halfZ + cb
        cA = np.tanh(cZ)
        cache["A"+str(current_l)]= cA
    cW = params["W"+str(L)]
    cb = params["b"+str(L)]
    AL = np.dot(cW,cA) + cb
    cache["A"+str(L)] = AL
    return cache,AL

def intialize_params(layer_size):
    layers = len(layer_size)
    parms = {}
    for i in range(1,layers):
        current_l = i
        parms["W"+str(current_l)] = np.random.rand(layer_size[i],layer_size[i-1]) * 1
        parms["b"+str(current_l)] = np.zeros((layer_size[i],1))
    return parms

def update_parm(grad,parms,alpha,layers):
    for i in range(1,layers):
        current_l = i
        parms["W"+str(current_l)] = parms["W"+str(current_l)] + alpha * grad["dW"+str(current_l)]
#         print(parms["b"+str(current_l)].shape)
#         print(grad["db"+str(current_l)].shape)
#         print((grad["db"+str(current_l)]+parms["b"+str(current_l)]).shape)
#         print("###########")

        parms["b"+str(current_l)] = parms["b"+str(current_l)] + alpha * grad["db"+str(current_l)]
def backpropag(cache,parms,layers,Y):
    grad = {}
    cA = cache["A"+str(layers-1)]
    #mid = np.multiply(cA,1-cA)
    mid = 1
    last = cache["A"+str(layers-2)]
    error = (Y - cA)
    delta = np.multiply(error,mid)

    grad["dW"+str(layers-1)] =  -np.dot(delta,last.T)
    ds = - np.sum(delta,axis = 1)
    grad["db"+str(layers-1)] = ds.reshape(ds.shape[0],1)
    for i in range(layers-2,0,-1):
        cW = parms["W"+str(i)]
        cW2 = parms["W"+str(i+1)]
        cb = parms["b"+str(i)]
        cA = cache["A"+str(i)]
        error = np.dot(cW2.T,delta)
        mid = np.multiply(cA,1-cA)
        last = cache["A"+str(i-1)]
        delta = np.multiply(error,mid)
        grad["dW"+str(i)] = - np.dot(delta,last.T)
        ds = - np.sum(delta,axis = 1)
        grad["db"+str(i)] = ds.reshape(ds.shape[0],1)



    return grad

def loss(a3,Y):
    dif = a3 - Y
    dif = dif * dif
    dif = np.sum(np.sum(dif,axis = 0))

    return dif / 2

# p = intialize_params(layer_size)
# for i in range(20):
#     print(i)
#     h = forward_propagation(p,X,layer_size)
#     s = backpropag(h,p,layers,Y)
#     update_parm(s,p,0.1,layers)
#     if(i % 1 == 0):
#         losss = loss(h["A2"],Y)
#         print("loss is : "+ str(losss))

def one_neuron_nn(X,Y):

    z = np.dot(X,w) + b
    a = sigmoid(z)

    d1 = -(Y - a)
    d2 = a * (1-a)
    d3 = X

    d =d1 * d2
    d =  np.dot(d3.T,d)


    # d = np.sum(d,axis = 0)
    # d = np.reshape(d,(d.shape[0],1))
    return a,d

def forward(X,w,b):
    z = np.dot(X,w) + b
    a = sigmoid(z)
    return a

w = np.random.rand(3,2)
b = np.zeros((1,1))
# xe = np.random.rand(3,256)
# ye = sigmoid(xe)
xe = np.random.rand(256,3)
ye = np.array([xe[:,0] + 2 * xe[:,1] - 3 * xe[:,2]  , -xe[:,0] * 10   ])
#pdb.set_trace()
ye =ye.T# np.reshape(ye,(10,2))
ye = sigmoid(ye)
import pdb

#pdb.set_trace()

for i in range(10000):
    a,d = one_neuron_nn(xe,ye)

    w = w -0.01 * d
    los = loss(a,ye)
    if(i % 1000 == 0):

        print("Loss is: {0}".format(los))


g = np.array([[1,-10],[2,0],[-3,-0]])
print(loss(ye,forward(xe,g,b)))

# p = intialize_params(layer_size)
# h,a2 = forward_propagation(p,X,layer_size)
# s = backpropag(h,p,layers,Y)
# print(a2,Y)
# loss(a2,Y)
# dw2 = s["dW2"]
# print(dw2)
